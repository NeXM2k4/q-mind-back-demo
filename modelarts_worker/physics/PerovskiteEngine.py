from modelarts_worker.mindspore_config import get_logger
import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops

logger = get_logger(__name__)

class PerovskiteEngine(Cell):
    """
    PerovskiteEngine: Physical modeling engine for hybrid halide perovskite
    MAPb(I(1-x)Br(x))3.

    Unlike BrusEngine/PbSEngine, the control variable is chemical composition
    (bromide molar fraction), not quantum dot size. The domain is bounded to
    x <= 0.20 by the photoinduced halide segregation boundary (Hoke effect):
    the mapping from gene to x makes unstable compositions structurally
    unreachable, so no penalty logic is needed in the optimizer.

    Source: ESPECS.pdf, "Especificación de física — Módulo de perovskita".
    """

    X_MIN, X_MAX = 0.0, 0.20
    EG_I, EG_BR, BOWING = 1.58, 2.28, 0.33
    DEDT, T_REF = 3.0e-4, 300.0
    E_URBACH, SIGMA_E = 0.014, 0.05
    VOC_OFFSET = 0.57
    FF = 0.78
    EQE = 0.85
    ALPHA_MAX = 1.0e7
    W_PEAK, W_CONT = 0.3, 0.7

    def __init__(self, max_absorption_coefficient: float = 1e7):
        super().__init__()
        self.alpha_max = max_absorption_coefficient
        self.w_peak, self.w_cont = self.W_PEAK, self.W_CONT
        self.ff = self.FF
        self.eqe = self.EQE

        logger.debug(
            "PerovskiteEngine ready | x=[%.2f, %.2f]  Eg(x)=[%.4f, %.4f] eV  "
            "Voc_offset=%.2f  FF=%.2f  EQE=%.2f",
            self.X_MIN, self.X_MAX, self.EG_I, self.EG_BR - self.BOWING * self.X_MAX * (1 - self.X_MAX),
            self.VOC_OFFSET, self.FF, self.EQE,
        )

    def construct(self, temperature, gene, wavelengths):
        """
        Args:
            temperature (float): Operating temperature [K].
            gene (ms.Tensor): Unit control variable in [0, 1], mapped to bromide
                fraction x in [0, 0.20].
            wavelengths (ms.Tensor): A 1D tensor of wavelengths to evaluate [nm].

        Returns:
            Tuple[ms.Tensor, ms.Tensor, ms.Tensor]: (absorption_coefficient, e_g, v_oc)
        """
        # 1. gen -> composición
        x = self.X_MIN + gene * (self.X_MAX - self.X_MIN)
        # 2. bandgap por ley de Vegard con bowing
        e_g = self.EG_I * (1 - x) + self.EG_BR * x - self.BOWING * x * (1 - x)
        # 3. corrección térmica (coeficiente POSITIVO)
        e_g = e_g + self.DEDT * (temperature - self.T_REF)
        # 4. perfil de absorción, calculado en energía
        photon_energy = 1239.84 / wavelengths
        peak = ops.exp(-ops.pow(photon_energy - e_g, 2) / (2 * self.SIGMA_E ** 2))
        cont = 1.0 / (1.0 + ops.exp(-(photon_energy - e_g) / self.E_URBACH))
        absorption = self.alpha_max * (self.w_peak * peak + self.w_cont * cont)
        # 5. voltaje, específico de este material
        v_oc = ops.maximum(e_g - self.VOC_OFFSET, ops.zeros_like(e_g))
        return absorption, e_g, v_oc

    def to_physical(self, gene: float) -> float:
        """Maps a unit gene in [0, 1] to the bromide fraction x, outside the graph."""
        return self.X_MIN + gene * (self.X_MAX - self.X_MIN)
