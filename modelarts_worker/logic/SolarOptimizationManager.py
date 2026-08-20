from modelarts_worker.mindspore_config import get_logger
import math
import pandas as pd
import mindspore as ms
from modelarts_worker.physics.BrusEngine import BrusEngine
from modelarts_worker.physics.SolarPerformanceEvaluator import SolarPerformanceEvaluator
from modelarts_worker.logic.GeneticSolarOptimizer import GeneticSolarOptimizer
from mindspore import ops as _ops

logger = get_logger(__name__)


class SolarOptimizationManager:
    """
    SolarOptimizationManager: Orchestrator class that bridges material data
    with the MindSpore-based genetic optimization engine.

    Architecture: Controller / Factory Pattern
    Responsibility: Material data management, engine instantiation, and study execution.

    Materials can come from either source:
    - csv_path: legacy Brus-only catalog (raw Varshni/Brus parameters per row),
      used by the authenticated CRUD-backed /optimization/run route.
    - material_registry: a name -> MaterialSpec mapping (see MaterialRegistry.py),
      used by the public demo route to mix Brus/PbS/perovskite engines freely.
    """

    def __init__(self, csv_path=None, kappa=0.5, material_registry=None):
        """
        Initializes the material database and prepares a local cache for physics engines.

        Args:
            csv_path (str): Path to the CSV containing semiconductor physical constants.
                Ignored if material_registry is provided.
            kappa (float): Penalty coefficient for current mismatch in the fitness function.
            material_registry (dict): Optional name -> MaterialSpec mapping. Takes
                precedence over csv_path.
        """
        self.kappa = kappa
        self.engines_cache = {}

        if material_registry is not None:
            self.catalog = material_registry
        else:
            df = pd.read_csv(csv_path)
            self.catalog = df.set_index('Material').to_dict('index')
        logger.info("Material catalog loaded: %d materials", len(self.catalog))

        # Evaluator instance used for post-study metrics (PHE). kappa is irrelevant here.
        self._evaluator = SolarPerformanceEvaluator(kappa=kappa)

    def _compute_photon_harvesting_efficiency(self, absorption_tensor, wavelengths_tensor) -> list:
        """
        Fraction of incident photons absorbed by each layer at the champion configuration.

        Uses the same Beer-Lambert inter-layer attenuation as SolarPerformanceEvaluator
        so that reported PHE is consistent with the fitness calculation.

        Args:
            absorption_tensor: MindSpore Tensor, shape (Layers, Wavelengths).
            wavelengths_tensor: MindSpore Tensor, the wavelength grid [nm].

        Returns:
            List[float]: PHE values in [0, 1], one per layer, rounded to 6 d.p.
        """
        photon_flux = self._evaluator._interpolate_spectrum(wavelengths_tensor)  # (W,)
        delta       = wavelengths_tensor[1] - wavelengths_tensor[0]
        total_flux  = (photon_flux * delta).sum()
        thickness   = self._evaluator.thickness
        n_layers    = absorption_tensor.shape[0]

        phe = []
        for i in range(n_layers):
            # Cumulative optical depth from layers above (0..i-1)
            if i == 0:
                attenuation = _ops.ones_like(photon_flux)                           # (W,)
            else:
                cumulative_tau = (absorption_tensor[:i] * thickness).sum(axis=0)   # (W,)
                attenuation   = _ops.exp(-cumulative_tau)                           # (W,)

            # Exact Beer-Lambert absorbed fraction in this layer: 1 − exp(−τ_i)
            tau_i            = absorption_tensor[i] * thickness                    # (W,)
            absorbed_fraction = 1.0 - _ops.exp(-tau_i)                            # (W,)

            flux_i = photon_flux * attenuation                                     # (W,)
            phe_i  = (flux_i * absorbed_fraction * delta).sum() / total_flux
            phe.append(round(float(phe_i.asnumpy()), 6))

        return phe

    def _compute_bandgaps_and_controls(
        self,
        materials: list,
        winner_genes,          # ms.Tensor shape (num_layers,) – champion from GA
        temp_tensor,           # ms.Tensor float32 – operating temperature
        wavelengths_tensor,    # ms.Tensor float32 – wavelength grid
    ):
        """
        Extracts the temperature-corrected bandgap and the physical control value
        (radius, diameter, composition, ...) for each layer at the champion
        configuration, using the already-cached engine instances.

        Returns:
            Tuple[List[float], List[float]]: (bandgaps_eV, control_values), one
            entry per layer, rounded to 6 d.p.
        """
        bandgaps = []
        control_values = []
        for i, name in enumerate(materials):
            engine = self.get_engine(name)
            gene_value = float(winner_genes[i:i + 1].asnumpy().flat[0])
            gene = winner_genes[i:i + 1].unsqueeze(0)
            _, e_g, _ = engine(
                temperature=temp_tensor,
                gene=gene,
                wavelengths=wavelengths_tensor,
            )
            bandgaps.append(round(float(e_g.asnumpy().flat[0]), 6))
            control_values.append(round(engine.to_physical(gene_value), 6))
        return bandgaps, control_values

    def get_engine(self, name):
        """
        Retrieves or instantiates a physics engine for a specific material.

        Args:
            name (str): The material identifier (e.g., 'PbS', 'CdSe').

        Returns:
            Cell: The physics-informed MindSpore cell for the requested material.
        """
        if name not in self.engines_cache:
            entry = self.catalog[name]
            if hasattr(entry, 'build_engine'):
                self.engines_cache[name] = entry.build_engine()
            else:
                self.engines_cache[name] = BrusEngine(
                    bandgap=entry['Eg_0K_eV'],
                    alpha=entry['Alpha_evK'],
                    beta=entry['Beta_K'],
                    me_eff=entry['me_eff'],
                    mh_eff=entry['mh_eff'],
                    eps_r=entry['epsilon_r']
                )
        return self.engines_cache[name]

    def run_study(self, user_params):
        """
        Executes a complete evolutionary optimization study based on user-defined parameters.

        Args:
            user_params (dict): Configuration dictionary with the following keys:
                - 'materials'(list[str]): Material names from the catalog.
                - 'pop_size'(int): Population size for the genetic algorithm.
                - 'alpha'(float): Arithmetic crossover weight [0.0 - 1.0].
                - 'mutation'(float): Gaussian mutation strength [0.0 - 1.0].
                - 'iterations'(int): Number of generations to evolve.
                - 'temp'(ms.Tensor float32) : Temperature in Kelvin.
                - 'wavelength' (ms.Tensor float32) : Wavelength range in nm.

        Returns:
            dict: fitness_history, pce_history, avg_fitness_history, best_genes,
            absorption_spectrum, bandgaps_eV, control_values, champion_cmi,
            champion_voc, champion_jsc, phe.
        """
        selected_engines = [self.get_engine(m) for m in user_params['materials']]

        optimizer = GeneticSolarOptimizer(
            engines=selected_engines,
            population_size=user_params['pop_size'],
            num_layers=len(user_params['materials']),
            alpha=user_params['alpha'],
            mutation_strength=user_params['mutation'],
            kappa=user_params.get('kappa', self.kappa)
        )

        materials_str = ', '.join(user_params['materials'])
        logger.info(
            "Study START | materials=[%s]  pop=%d  iters=%d  κ=%.3f  T=%.1f K",
            materials_str,
            user_params['pop_size'],
            user_params['iterations'],
            user_params.get('kappa', self.kappa),
            float(user_params['temp'].asnumpy().flat[0]),
        )

        _LOG_EVERY = max(1, user_params['iterations'] // 10)  # log ~10 checkpoints

        fitness_results = []
        pce_results = []
        avg_fitness_results = []
        try:
            for i in range(user_params['iterations']):
                fitness, pce, cmi, winner, absorption_tensor, avg_fitness, voc, jsc = optimizer(
                    user_params['temp'], user_params['wavelength']
                )  # type: ignore

                f_val   = float(fitness.asnumpy())     # type: ignore
                pce_val = float(pce.asnumpy())         # type: ignore
                avg_val = round(float(avg_fitness.asnumpy()), 6)  # type: ignore

                # NaN / Inf guard — surface silent MindSpore graph failures immediately
                if math.isnan(f_val) or math.isinf(f_val):
                    logger.error(
                        "Gen %d/%d: fitness=%s  pce=%.6f  avg=%.6f — NaN/Inf detected! "
                        "Check absorption and bandgap values.",
                        i + 1, user_params['iterations'], f_val, pce_val, avg_val,
                    )
                elif math.isnan(avg_val) or math.isinf(avg_val):
                    logger.warning(
                        "Gen %d/%d: fitness=%.6f  pce=%.6f  avg_fitness=%s — "
                        "avg NaN/Inf (some individuals may be degenerate).",
                        i + 1, user_params['iterations'], f_val, pce_val, avg_val,
                    )
                elif (i + 1) % _LOG_EVERY == 0 or i == 0:
                    logger.info(
                        "Gen %d/%d: best_fitness=%.6f  pce=%.4f%%  avg_fitness=%.6f",
                        i + 1, user_params['iterations'],
                        f_val, pce_val * 100, avg_val,
                    )

                fitness_results.append(f_val)
                pce_results.append(pce_val)
                avg_fitness_results.append(avg_val)

        except Exception:
            logger.exception(
                "Study FAILED at generation %d/%d for materials=[%s]",
                i + 1, user_params['iterations'], materials_str,
            )
            raise

        absorption_spectrum = absorption_tensor.asnumpy().tolist()  # type: ignore
        champion_cmi = round(float(cmi.asnumpy()), 6)  # type: ignore
        champion_voc = round(float(voc.asnumpy()), 6)  # type: ignore
        champion_jsc = round(float(jsc.asnumpy()), 6)  # type: ignore
        phe = self._compute_photon_harvesting_efficiency(absorption_tensor, user_params['wavelength'])

        bandgaps_eV, control_values = self._compute_bandgaps_and_controls(
            user_params['materials'], winner, user_params['temp'], user_params['wavelength']
        )

        logger.info(
            "Study END | best_pce=%.4f%%  cmi=%.6f  voc=%.4f V  jsc=%.4f A/m²  bandgaps=%s",
            pce_results[-1] * 100 if pce_results else float('nan'),
            champion_cmi, champion_voc, champion_jsc, bandgaps_eV,
        )

        return {
            "fitness_history": fitness_results,
            "pce_history": pce_results,
            "avg_fitness_history": avg_fitness_results,
            "best_genes": winner,
            "absorption_spectrum": absorption_spectrum,
            "bandgaps_eV": bandgaps_eV,
            "control_values": control_values,
            "champion_cmi": champion_cmi,
            "champion_voc": champion_voc,
            "champion_jsc": champion_jsc,
            "champion_ff": optimizer.ff,
            "phe": phe,
        }
