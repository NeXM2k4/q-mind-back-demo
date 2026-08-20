from pydantic import BaseModel, Field, model_validator
from typing import List


class SimulationRequest(BaseModel):
    """
    Schema for the demo's public solar cell simulation request.

    Only exposes the three variables the demo lets a visitor change: which
    two materials form the tandem stack, the operating temperature, and the
    AM1.5G spectral crop window. Every other genetic-algorithm and physical
    parameter is fixed server-side and never exposed.
    """

    materials: List[str] = Field(
        ...,
        min_length=2,
        max_length=2,
        examples=[["Perovskita yoduro bromuro", "CdSe"]],
        description="Exactly two materials for the tandem stack.",
    )

    temperature_c: float = Field(
        ...,
        ge=0.0,
        le=40.0,
        description="Operating temperature in Celsius.",
    )

    spectral_min_nm: float = Field(
        ...,
        ge=300.0,
        le=1400.0,
        description="Lower bound of the AM1.5G spectral crop window in nm.",
    )

    spectral_max_nm: float = Field(
        ...,
        ge=300.0,
        le=1400.0,
        description="Upper bound of the AM1.5G spectral crop window in nm.",
    )

    @model_validator(mode="after")
    def check_materials_and_range(self) -> "SimulationRequest":
        if self.materials[0] == self.materials[1]:
            raise ValueError("Los dos materiales deben ser distintos.")
        if self.spectral_max_nm <= self.spectral_min_nm:
            raise ValueError("spectral_max_nm debe ser mayor que spectral_min_nm.")
        return self


class LayerResult(BaseModel):
    """Physical result for a single layer of the tandem stack."""

    material: str
    bandgap_ev: float
    control_label: str = Field(description="Human-readable name of this material's physical control variable.")
    control_value: float = Field(description="Optimal value of the control variable, in control_unit.")
    control_unit: str
    thickness_nm: float
    absorption_m_inv: List[float] = Field(description="Absorption coefficient [m^-1] at each spectrum.wavelengths_nm point.")


class SpectrumResult(BaseModel):
    """AM1.5G reference spectrum over the requested crop window."""

    wavelengths_nm: List[float]
    irradiance_w_m2_nm: List[float]


class MetricsResult(BaseModel):
    """Secondary reported metrics."""

    fill_factor: float
    current_mismatch_index: float
    photon_harvesting_efficiency: List[float] = Field(description="Fraction of incident photons absorbed per layer, in materials order.")


class ConvergenceResult(BaseModel):
    """Genetic algorithm convergence trace."""

    pce_history: List[float]
    generations_to_convergence: int
    total_generations: int


class SimulationResponse(BaseModel):
    """Schema for the demo's public solar cell simulation response."""

    status: str = "COMPLETED"
    efficiency_pct: float = Field(description="Projected Power Conversion Efficiency, in percent.")
    voltage_v: float = Field(description="Series-summed open-circuit voltage of the tandem stack.")
    current_ma_cm2: float = Field(description="Current-matching-limited short-circuit current density.")
    computation_time_ms: float
    layers: List[LayerResult]
    spectrum: SpectrumResult
    metrics: MetricsResult
    convergence: ConvergenceResult
