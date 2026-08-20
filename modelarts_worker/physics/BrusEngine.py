from modelarts_worker.mindspore_config import get_logger
import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops

logger = get_logger(__name__)

class BrusEngine(Cell):
    """
    BrusEngine: A physical modeling engine for Quantum Dot (QD) semiconductors.

    This class implements the Brus Equation to calculate the size-dependent energy
    gap of nanoparticles, incorporating Varshni's Law for temperature correction
    and a peak+continuum absorption profile evaluated in energy space.

    The control variable is the QD radius, mapped from a unit gene in [0, 1]
    so this engine shares a common interface with composition-controlled
    engines (e.g. PerovskiteEngine) inside the same tandem stack.

    Architecture: Modular / Decoupled
    Framework: MindSpore (Optimized for GPU/NPU acceleration)
    """

    def __init__(self,
                 bandgap: float,
                 alpha: float,
                 beta: float,
                 me_eff: float,
                 mh_eff: float,
                 eps_r: float,
                 max_absorption_coefficient: float = 1e7,
                 r_min: float = 2.0,
                 r_max: float = 10.0,
                 voc_offset: float = 0.4,
                 sigma_e: float = 0.05,
                 e_urbach: float = 0.014,
                 w_peak: float = 0.3,
                 w_cont: float = 0.7,
                 ff: float = 0.75,
                 eqe: float = 1.0):
        """
        Initializes the material-specific intrinsic properties.

        Args:
            bandgap (float): Bulk energy gap (E0) at 0 Kelvin [eV].
            alpha (float): Varshni thermal coefficient [eV/K].
            beta (float): Varshni constant related to Debye temperature [K].
            me_eff (float): Effective mass of the electron [relative to m0].
            mh_eff (float): Effective mass of the hole [relative to m0].
            eps_r (float): Relative dielectric constant of the material [dimensionless].
            max_absorption_coefficient (float): Peak absorption value [m^-1]. Default is 1e7.
            r_min (float): Minimum QD radius reachable by gene=0 [nm].
            r_max (float): Maximum QD radius reachable by gene=1 [nm].
            voc_offset (float): V_oc deficit vs. bandgap, V_oc = max(E_g - voc_offset, 0) [V].
            sigma_e (float): Excitonic peak width [eV].
            e_urbach (float): Urbach tail width of the continuum absorption onset [eV].
            w_peak (float): Weight of the excitonic peak term in the absorption profile.
            w_cont (float): Weight of the continuum term in the absorption profile.
            ff (float): Fill factor of this material's layer [dimensionless].
            eqe (float): External quantum efficiency of this material's layer [dimensionless].
        """
        super().__init__()

        # Material properties
        self.bandgap = bandgap
        self.alpha = alpha
        self.beta = beta
        self.me_eff = me_eff
        self.mh_eff = mh_eff
        self.eps_r = eps_r
        self.max_absorption_coefficient = max_absorption_coefficient

        # Gene -> radius mapping bounds
        self.r_min = r_min
        self.r_max = r_max

        # Device-level reporting properties (Python-side, not part of the graph)
        self.voc_offset = voc_offset
        self.ff = ff
        self.eqe = eqe

        # Pre-scaled physical constants (optimized for eV and nm units)
        # Avoids numerical underflow in float32 operations
        self.BRUS_CONST = 0.3760   # Confinement constant: h^2 / (8*m0) in [eV·nm^2]
        self.COUL_CONST = 2.5682   # Coulomb constant: 1.786*q / (4*pi*eps0) in [eV·nm]

        # Absorption profile: excitonic peak (Gaussian) + continuum (sigmoid), in energy space
        self.sigma_e = sigma_e
        self.e_urbach = e_urbach
        self.w_peak = w_peak
        self.w_cont = w_cont

        logger.debug(
            "BrusEngine ready | Eg0=%.4f eV  α=%.2e  β=%.1f K  "
            "me*=%.3f  mh*=%.3f  εr=%.2f  r=[%.2f, %.2f] nm  α_max=%.2e m⁻¹",
            bandgap, alpha, beta, me_eff, mh_eff, eps_r, r_min, r_max, max_absorption_coefficient,
        )

    def construct(self, temperature: float, gene: ms.Tensor, wavelengths: ms.Tensor) -> ms.Tensor:
        """
        Computes the absorption coefficient spectrum based on the QD size and temperature.

        Args:
            temperature (float): Operating temperature [K].
            gene (ms.Tensor): Unit control variable in [0, 1], mapped to QD radius [nm].
            wavelengths (ms.Tensor): A 1D tensor of wavelengths to evaluate [nm].

        Returns:
            Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
                - absorption_coefficient: profile [m^-1] for the given wavelength range.
                - e_qd: size- and temperature-corrected bandgap [eV].
                - v_oc: open-circuit voltage of this layer [V].
        """
        radius = self.r_min + gene * (self.r_max - self.r_min)

        # Varshni's Law: Temperature-dependent bandgap correction
        # Formula: Eg(T) = E0 - (alpha * T^2) / (T + beta)
        e_bulk = self.bandgap - (self.alpha * ops.pow(temperature, 2)) / (temperature + self.beta)

        # Brus Equation: Quantum confinement and Coulomb interaction (in eV and nm)
        confinement = (self.BRUS_CONST / ops.pow(radius, 2)) * (1/self.me_eff + 1/self.mh_eff)

        # Coulomb term: decreases energy gap due to electron-hole attraction
        coulomb = self.COUL_CONST / (self.eps_r * radius)

        # Total quantum dot energy gap [eV]
        e_qd = e_bulk + confinement - coulomb

        # Absorption profile, evaluated in energy space: excitonic peak + continuum above the gap
        photon_energy = 1239.84 / wavelengths
        peak = ops.exp(-ops.pow(photon_energy - e_qd, 2) / (2 * self.sigma_e ** 2))
        cont = 1.0 / (1.0 + ops.exp(-(photon_energy - e_qd) / self.e_urbach))
        absorption_coefficient = self.max_absorption_coefficient * (self.w_peak * peak + self.w_cont * cont)

        v_oc = ops.maximum(e_qd - self.voc_offset, ops.zeros_like(e_qd))

        return absorption_coefficient, e_qd, v_oc

    def to_physical(self, gene: float) -> float:
        """Maps a unit gene in [0, 1] to the QD radius [nm], outside the graph."""
        return self.r_min + gene * (self.r_max - self.r_min)
