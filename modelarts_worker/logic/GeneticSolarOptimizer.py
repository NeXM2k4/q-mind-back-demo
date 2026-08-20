from modelarts_worker.mindspore_config import get_logger
from mindspore.nn import Cell
from mindspore import ops, Tensor, dtype
from modelarts_worker.physics.SolarPerformanceEvaluator import SolarPerformanceEvaluator
import mindspore as ms
from typing import List, Tuple

logger = get_logger(__name__)

class GeneticSolarOptimizer(Cell):
    """
    GeneticSolarOptimizer: A MindSpore-based evolutionary engine for QD/perovskite solar cells.

    This class performs batch optimization of multi-layer tandem configurations
    using tournament selection, arithmetic crossover, and Gaussian mutation.

    The population lives in a unit hypercube [0, 1]^(pop_size, num_layers): the
    optimizer never sees physical units. Each layer's material engine maps its
    own gene column to its own physical control variable (radius, diameter,
    composition, ...), so materials tuned by different physical quantities can
    share the same population and be evolved together.
    """

    def __init__(self, population_size: int = 100, num_layers: int = 2, engines: list = None, alpha: float = 0.5, mutation_strength: float = 0.1, kappa: float = 0.5):
        super().__init__()
        # Unit hypercube bounds for the gene space
        self.g_min = Tensor(0.0, ms.float32)
        self.g_max = Tensor(1.0, ms.float32)

        # Performance evaluator (Efficiency + Current Mismatch Penalty)
        self.evaluator = SolarPerformanceEvaluator(kappa=kappa)
        self.engines = engines if engines is not None else []
        self.alpha = alpha
        self.mutation_strength = mutation_strength

        # Material-fixed reporting properties (Python-side, not part of the graph):
        # EQE varies per layer and broadcasts against the (Batch, Layers, W) tensors;
        # FF is the device fill factor, taken as the minimum across layers (current-limited).
        self.eqe = Tensor([getattr(e, 'eqe', 1.0) for e in self.engines], ms.float32)
        self.ff = min(getattr(e, 'ff', 0.75) for e in self.engines) if self.engines else 0.75

        # Population Initialization: Random uniform distribution over the unit hypercube
        # Shape: (population_size, num_layers)
        self.population = ms.Parameter(ops.uniform(
            (population_size, num_layers),
            self.g_min,
            self.g_max,
            dtype=ms.float32
        ), name="population")

        logger.info(
            "GeneticSolarOptimizer ready | pop=%d  layers=%d  α=%.2f  "
            "mutation=%.3f  κ=%.3f  ff=%.3f  gene=[0, 1]",
            population_size, num_layers, alpha, mutation_strength, kappa, self.ff,
        )

    def construct(self, temperature, wavelength) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        absorption_list = []
        e_g_list = []
        v_oc_list = []

        # Spectral Response Calculation: Map each engine to its corresponding layer
        for i, engine in enumerate(self.engines):
            # Extract genes for the specific layer [Batch, 1]
            gene_layer = self.population[:, i:i+1]

            # Physical modeling of absorption, energy gap, and open-circuit voltage
            abs_layer, e_g_layer, v_oc_layer = engine(
                temperature = temperature,
                gene = gene_layer,
                wavelengths = wavelength
            )
            absorption_list.append(abs_layer)
            e_g_list.append(e_g_layer)
            v_oc_list.append(v_oc_layer)

        # Vectorized assembly: Shape (Batch, Layers, Wavelengths)
        absortion_batch = ops.stack(absorption_list, axis=1)
        e_g_batch = ops.stack(e_g_list, axis=1).squeeze(axis=-1)
        v_oc_batch = ops.stack(v_oc_list, axis=1).squeeze(axis=-1)

        # Fitness Evaluation: PCE - kappa * Sum(|Ji - Ji+1|)
        fitness_batch, efficiency_batch, cmi_batch, v_oc_total_batch, j_sc_limit_batch = self.evaluator(
            absorption_coefficient = absortion_batch,
            v_oc = v_oc_batch,
            eqe = self.eqe,
            wavelengths = wavelength,
            ff = self.ff
        )

        # Elitism: Identify the best performing individual (Champion)
        winner = fitness_batch.argmax()
        winner_genes = self.population[winner,:]

        # Tournament Selection: Pairwise competition
        # Formula: Select Parent if Fitness(A) > Fitness(B)
        candidates = ops.randint(low=0, high=self.population.shape[0], size=(self.population.shape[0], 2), dtype=ms.int32)
        selected_parents_indices = ops.where(
            fitness_batch[candidates[:,0]] > fitness_batch[candidates[:,1]],
            candidates[:,0],
            candidates[:,1]
        )
        parents_genes = self.population[selected_parents_indices]

        # Arithmetic Crossover: Weighted average of parent genes
        # Formula: Offspring = alpha * P1 + (1 - alpha) * P2
        pop_size = self.population.shape[0]
        half_pop = pop_size // 2

        p1 = parents_genes[:half_pop, :]
        p2 = parents_genes[half_pop:half_pop*2, :]
        offspring = self.alpha * p1 + (1 - self.alpha) * p2

        # Gaussian Mutation: Stochastic search with boundary clipping
        # Formula: G_new = Clip(G_old + N(0, sigma), g_min, g_max)
        offspring = offspring + ops.standard_normal(p1.shape) * self.mutation_strength
        offspring = ops.clip_by_value(offspring, self.g_min, self.g_max)

        # Generational Update: Combine Elite + Offspring + Survived Parents
        survivors_count = pop_size - half_pop - 1  # Remaining slots after offspring and elite
        new_population = ops.concat((offspring, winner_genes.expand_dims(axis=0), parents_genes[:survivors_count]))
        ops.assign(self.population, new_population)

        avg_fitness = fitness_batch.mean()
        return (
            fitness_batch[winner], efficiency_batch[winner], cmi_batch[winner],
            winner_genes, absortion_batch[winner], avg_fitness,
            v_oc_total_batch[winner], j_sc_limit_batch[winner],
        )  #type: ignore
