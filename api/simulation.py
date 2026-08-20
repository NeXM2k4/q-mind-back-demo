import asyncio
import time

from fastapi import APIRouter, HTTPException, status

from db.schemas.simulation import (
    SimulationRequest,
    SimulationResponse,
    LayerResult,
    SpectrumResult,
    MetricsResult,
    ConvergenceResult,
)
from modelarts_worker.logic.DataAnalyzer import DataAnalyzer
from modelarts_worker.logic.MaterialRegistry import DEMO_MATERIALS, is_valid_material, available_materials
from modelarts_worker.mindspore_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["simulation"])

DEMO_KAPPA = 0.5
DEMO_POPULATION_SIZE = 60
DEMO_MAX_ITERATIONS = 40
DEMO_CROSSOVER_ALPHA = 0.5
DEMO_MUTATION_STRENGTH = 0.1
DEMO_WAVELENGTH_STEP_NM = 10.0

DEMO_CONCURRENCY_LIMIT = 1  # MindSpore's graph compiler is not thread-safe; only one study may compile/run at a time
AM_TO_MA_CM2 = 0.1  # 1 A/m^2 = 0.1 mA/cm^2

_analyzer = DataAnalyzer(kappa=DEMO_KAPPA, material_registry=DEMO_MATERIALS)
_simulation_semaphore = asyncio.Semaphore(DEMO_CONCURRENCY_LIMIT)


def _celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15


def run_simulation(
    materials,
    temperature_c: float,
    spectral_min_nm: float,
    spectral_max_nm: float,
    population_size: int = DEMO_POPULATION_SIZE,
    max_iterations: int = DEMO_MAX_ITERATIONS,
):
    request = {
        "materials": materials,
        "operating_temperature": _celsius_to_kelvin(temperature_c),
        "population_size": population_size,
        "max_iterations": max_iterations,
        "crossover_alpha": DEMO_CROSSOVER_ALPHA,
        "mutation_strength": DEMO_MUTATION_STRENGTH,
        "kappa": DEMO_KAPPA,
        "wavelength_input_csv": False,
        "wavelength_left_bound": spectral_min_nm,
        "wavelength_right_bound": spectral_max_nm,
        "wavelength_step": DEMO_WAVELENGTH_STEP_NM,
    }
    return _analyzer.analyze(request)


def warmup() -> None:
    """
    Forces MindSpore graph compilation once at startup, on a tiny population,
    so the first real visitor at the conference doesn't pay a >10s compile cost.
    """
    logger.info("Warming up the simulation engine (MindSpore graph compilation)...")
    try:
        materials = available_materials()[:2]
        run_simulation(materials, 25.0, 300.0, 1400.0, population_size=5, max_iterations=1)
        logger.info("Warmup complete.")
    except Exception:
        logger.exception("Warmup failed; the first real request will pay the compilation cost.")


def _build_response(materials, result: dict, elapsed_ms: float) -> SimulationResponse:
    layers = [
        LayerResult(
            material=name,
            bandgap_ev=result["bandgaps_eV"][i],
            control_label=DEMO_MATERIALS[name].control_label,
            control_value=result["optimal_control_values"][i],
            control_unit=DEMO_MATERIALS[name].control_unit,
            thickness_nm=300.0,
            absorption_m_inv=result["absorption_spectrum"][i],
        )
        for i, name in enumerate(materials)
    ]

    return SimulationResponse(
        efficiency_pct=round(result["projected_pce"] * 100, 4),
        voltage_v=result["voltage_v"],
        current_ma_cm2=round(result["current_a_m2"] * AM_TO_MA_CM2, 4),
        computation_time_ms=round(elapsed_ms, 2),
        layers=layers,
        spectrum=SpectrumResult(
            wavelengths_nm=result["wavelengths_nm"],
            irradiance_w_m2_nm=result["irradiance_w_m2_nm"],
        ),
        metrics=MetricsResult(
            fill_factor=result["fill_factor"],
            current_mismatch_index=result["current_mismatch_index"],
            photon_harvesting_efficiency=result["photon_harvesting_efficiency"],
        ),
        convergence=ConvergenceResult(
            pce_history=result["pce_history"],
            generations_to_convergence=result["generations_to_convergence"],
            total_generations=DEMO_MAX_ITERATIONS,
        ),
    )


@router.post("/simulate", response_model=SimulationResponse, status_code=status.HTTP_200_OK)
async def simulate(request: SimulationRequest) -> SimulationResponse:
    """
    Runs the tandem solar cell optimization for the demo's three exposed
    variables (materials, temperature, spectral window). Every genetic
    algorithm and physical parameter beyond those is fixed server-side.
    """
    invalid = [m for m in request.materials if not is_valid_material(m)]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Materiales no reconocidos: {invalid}. Disponibles: {available_materials()}",
        )

    logger.info(
        "Simulation request | materials=%s  T=%.1f C  range=[%.0f, %.0f] nm",
        request.materials, request.temperature_c, request.spectral_min_nm, request.spectral_max_nm,
    )

    async with _simulation_semaphore:
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                run_simulation,
                request.materials,
                request.temperature_c,
                request.spectral_min_nm,
                request.spectral_max_nm,
            )
        except Exception:
            logger.exception("Simulation failed | materials=%s", request.materials)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="La simulación falló. Intenta con otra combinación de materiales.",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Simulation DONE | materials=%s  time=%.0f ms  pce=%.4f%%",
        request.materials, elapsed_ms, result.get("projected_pce", 0) * 100,
    )

    return _build_response(request.materials, result, elapsed_ms)
