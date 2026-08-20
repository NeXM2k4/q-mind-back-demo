from modelarts_worker.mindspore_config import get_logger
import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops

logger = get_logger(__name__)

class PbSEngine(Cell):
    """
    PbSEngine: Physical modeling engine for PbS quantum dot semiconductors.

    Unlike BrusEngine, the size-dependent bandgap uses the empirical Moreels
    relation instead of the analytic Brus equation, since PbS's small effective
    masses and large dielectric screening make the Brus confinement + Coulomb
    terms diverge outside the range this model is validated for
    (see ESPECS.pdf §10). Temperature correction reuses PbS's own Varshni
    parameters (confirmed_materials.csv), evaluated as a pure thermal shift on
    top of the Moreels 300K baseline, mirroring PerovskiteEngine's
    composition-term + thermal-term structure.

    The control variable is the QD diameter, mapped from a unit gene in [0, 1],
    bounded to the Moreels-validated range (3.9-13.3 nm).
    """

    D_MIN, D_MAX = 3.9, 13.3

    MOREELS_EG0 = 0.41
    MOREELS_A = 0.0252
    MOREELS_B = 0.283

    VARSHNI_EG0, VARSHNI_ALPHA, VARSHNI_BETA = 0.286, -0.00041, 145.0
    T_REF = 300.0

    VOC_SLOPE, VOC_OFFSET = 0.519, 0.0221

    SIGMA_E, E_URBACH = 0.05, 0.014
    W_PEAK, W_CONT = 0.3, 0.7

    def __init__(self, max_absorption_coefficient: float = 1e7, ff: float = 0.75, eqe: float = 1.0):
        super().__init__()
        self.alpha_max = max_absorption_coefficient
        self.ff = ff
        self.eqe = eqe
        self.e_bulk_ref = self.VARSHNI_EG0 - self.VARSHNI_ALPHA * (self.T_REF ** 2) / (self.T_REF + self.VARSHNI_BETA)

        logger.debug(
            "PbSEngine ready | d=[%.1f, %.1f] nm  Voc=%.3f*Eg-%.4f  α_max=%.2e m⁻¹",
            self.D_MIN, self.D_MAX, self.VOC_SLOPE, self.VOC_OFFSET, max_absorption_coefficient,
        )

    def construct(self, temperature: float, gene: ms.Tensor, wavelengths: ms.Tensor) -> ms.Tensor:
        """
        Args:
            temperature (float): Operating temperature [K].
            gene (ms.Tensor): Unit control variable in [0, 1], mapped to QD diameter [nm].
            wavelengths (ms.Tensor): A 1D tensor of wavelengths to evaluate [nm].

        Returns:
            Tuple[ms.Tensor, ms.Tensor, ms.Tensor]: (absorption_coefficient, e_g, v_oc)
        """
        d = self.D_MIN + gene * (self.D_MAX - self.D_MIN)
        e_g_300 = self.MOREELS_EG0 + 1.0 / (self.MOREELS_A * ops.pow(d, 2) + self.MOREELS_B * d)

        e_bulk_t = self.VARSHNI_EG0 - self.VARSHNI_ALPHA * ops.pow(temperature, 2) / (temperature + self.VARSHNI_BETA)
        e_g = e_g_300 + (e_bulk_t - self.e_bulk_ref)

        photon_energy = 1239.84 / wavelengths
        peak = ops.exp(-ops.pow(photon_energy - e_g, 2) / (2 * self.SIGMA_E ** 2))
        cont = 1.0 / (1.0 + ops.exp(-(photon_energy - e_g) / self.E_URBACH))
        absorption = self.alpha_max * (self.W_PEAK * peak + self.W_CONT * cont)

        v_oc = ops.maximum(self.VOC_SLOPE * e_g - self.VOC_OFFSET, ops.zeros_like(e_g))

        return absorption, e_g, v_oc

    def to_physical(self, gene: float) -> float:
        """Maps a unit gene in [0, 1] to the QD diameter [nm], outside the graph."""
        return self.D_MIN + gene * (self.D_MAX - self.D_MIN)
