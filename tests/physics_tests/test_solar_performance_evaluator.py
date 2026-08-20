import pytest
import numpy as np
import mindspore as ms
from modelarts_worker.physics.SolarPerformanceEvaluator import SolarPerformanceEvaluator


def _generic_voc(e_qd_np, offset=0.4):
    """Mirrors BrusEngine's default V_oc model, for building test inputs by hand."""
    return np.maximum(e_qd_np - offset, 0.0)


class TestSolarPerformanceEvaluator:

    def test_solar_spectrum_loading(self):
        """Validates that AM1.5G solar spectrum is loaded correctly."""
        evaluator = SolarPerformanceEvaluator()

        # Check photon flux
        flux_data = evaluator.photon_flux.asnumpy()
        assert flux_data.dtype == 'float32', "Photon flux should be float32"
        assert flux_data.size > 0, "Photon flux should not be empty"
        assert (flux_data >= 0).all(), "Photon flux should be non-negative"
        assert flux_data.mean() > 0, "Mean photon flux should be positive"

        # Check wavelengths
        wavelengths = evaluator.wavelengths.asnumpy()
        assert wavelengths.size == flux_data.size, "Wavelengths and flux must have same size"
        assert wavelengths.min() >= 280, "Min wavelength should be >= 280 nm"
        assert wavelengths.max() <= 4500, "Max wavelength should be <= 4500 nm"

        # Check total solar power (AM1.5G global irradiance), trapz-integrated over
        # the non-uniform reference grid. Standard AM1.5G is ~1000.37 W/m^2.
        p_sun = evaluator.p_sun.asnumpy().item()
        assert 950 < p_sun < 1050, f"Total solar power should be ~1000 W/m^2, got {p_sun:.1f}"

        print(f"Solar spectrum: {wavelengths.size} points, P_sun={p_sun:.1f} W/m^2")

    def test_single_layer_performance(self):
        """Tests performance calculation for a single-layer solar cell."""
        evaluator = SolarPerformanceEvaluator(kappa=0.5)

        # Create a single cell with uniform absorption
        num_wavelengths = evaluator.wavelengths.shape[0]
        absorption = ms.Tensor(np.ones((1, 1, num_wavelengths)) * 0.8, ms.float32)  # 80% absorption
        e_qd = np.array([[1.5]])  # 1.5 eV bandgap
        v_oc = ms.Tensor(_generic_voc(e_qd), ms.float32)
        eqe = ms.Tensor([1.0], ms.float32)

        fitness, efficiency, cmi, v_oc_total, j_sc_limit = evaluator(absorption, v_oc, eqe, evaluator.wavelengths, ff=0.75)

        assert fitness.shape == (1,), "Fitness should have shape (batch_size,)"
        assert efficiency.shape == (1,), "Efficiency should have shape (batch_size,)"
        assert cmi.shape == (1,), "CMI should have shape (batch_size,)"

        fitness_val = fitness.asnumpy().item()
        efficiency_val = efficiency.asnumpy().item()
        cmi_val = cmi.asnumpy().item()
        v_oc_val = v_oc_total.asnumpy().item()
        j_sc_val = j_sc_limit.asnumpy().item()

        assert not np.isnan(fitness_val), "Fitness should not be NaN"
        assert not np.isinf(fitness_val), "Fitness should not be Inf"
        # Single layer -> no mismatch penalty, fitness == efficiency
        assert np.isclose(fitness_val, efficiency_val, atol=1e-5), \
            "Single-layer fitness should equal efficiency (no mismatch penalty)"
        assert cmi_val == 0.0, "Single-layer CMI should be zero"
        assert -200 < fitness_val < 2.0, f"Fitness out of expected range, got {fitness_val:.4f}"
        assert v_oc_val == pytest.approx(1.1, abs=1e-5), "V_oc should equal the single layer's own V_oc"
        assert j_sc_val > 0, "J_sc should be positive for a fully illuminated absorbing layer"

        print(f"Single layer: fitness={fitness_val:.4f}, efficiency={efficiency_val:.4f}, cmi={cmi_val:.4f}, voc={v_oc_val:.4f}")

    def test_tandem_cell_performance(self):
        """Tests performance calculation for a 2-layer tandem solar cell."""
        evaluator = SolarPerformanceEvaluator(kappa=0.5)

        num_wavelengths = evaluator.wavelengths.shape[0]

        # Create 2-layer tandem: top layer (high Eg), bottom layer (low Eg)
        absorption = ms.Tensor(np.ones((1, 2, num_wavelengths)) * 0.7, ms.float32)
        e_qd = np.array([[1.8, 1.2]])  # Top: 1.8 eV, Bottom: 1.2 eV
        v_oc = ms.Tensor(_generic_voc(e_qd), ms.float32)
        eqe = ms.Tensor([1.0, 1.0], ms.float32)

        fitness, efficiency, cmi, v_oc_total, j_sc_limit = evaluator(absorption, v_oc, eqe, evaluator.wavelengths, ff=0.75)

        fitness_val = fitness.asnumpy().item()
        efficiency_val = efficiency.asnumpy().item()
        cmi_val = cmi.asnumpy().item()
        v_oc_val = v_oc_total.asnumpy().item()

        assert not np.isnan(fitness_val), "Tandem fitness should not be NaN"
        assert not np.isinf(fitness_val), "Tandem fitness should not be Inf"
        assert -200 < fitness_val < 2.0, f"Fitness out of expected range, got {fitness_val:.4f}"
        # Tandem with equal absorption -> some mismatch -> fitness < efficiency
        assert fitness_val <= efficiency_val, "Penalized fitness should not exceed raw efficiency"
        assert cmi_val >= 0.0, "CMI must be non-negative"
        assert v_oc_val == pytest.approx(1.4 + 0.8, abs=1e-5), "V_oc should be the series sum across layers"

        print(f"Tandem cell: fitness={fitness_val:.4f}, efficiency={efficiency_val:.4f}, cmi={cmi_val:.4f}, voc={v_oc_val:.4f}")

    def test_current_matching_penalty(self):
        """Validates that current mismatch reduces fitness."""
        evaluator = SolarPerformanceEvaluator(kappa=0.5)

        num_wavelengths = evaluator.wavelengths.shape[0]
        eqe = ms.Tensor([1.0, 1.0], ms.float32)
        e_qd = np.array([[1.5, 1.5]])
        v_oc = ms.Tensor(_generic_voc(e_qd), ms.float32)

        # Matched currents: both layers absorb equally
        absorption_matched = ms.Tensor(np.ones((1, 2, num_wavelengths)) * 0.8, ms.float32)
        fitness_matched, efficiency_matched, cmi_matched, _, _ = evaluator(absorption_matched, v_oc, eqe, evaluator.wavelengths, ff=0.75)
        fitness_matched_val = fitness_matched.asnumpy().item()
        cmi_matched_val = cmi_matched.asnumpy().item()

        absorption_mismatched = ms.Tensor(np.concatenate([
            np.ones((1, 1, num_wavelengths)) * 0.3,  # layer 0: weak -> j_min
            np.ones((1, 1, num_wavelengths)) * 0.9,  # layer 1: strong -> diff > 0
        ], axis=1), ms.float32)
        fitness_mismatched, _, cmi_mismatched, _, _ = evaluator(absorption_mismatched, v_oc, eqe, evaluator.wavelengths, ff=0.75)
        fitness_mismatched_val = fitness_mismatched.asnumpy().item()
        cmi_mismatched_val = cmi_mismatched.asnumpy().item()

        assert fitness_matched_val > fitness_mismatched_val, \
            f"Matched currents should have higher fitness: {fitness_matched_val:.4f} vs {fitness_mismatched_val:.4f}"
        assert cmi_mismatched_val > cmi_matched_val, \
            f"Mismatched config should have higher CMI: {cmi_mismatched_val:.4f} vs {cmi_matched_val:.4f}"

        print(f"Current matching: matched={fitness_matched_val:.4f} (cmi={cmi_matched_val:.4f}) > mismatched={fitness_mismatched_val:.4f} (cmi={cmi_mismatched_val:.4f})")

    def test_batch_processing(self):
        """Validates that evaluator can process multiple designs simultaneously."""
        evaluator = SolarPerformanceEvaluator(kappa=0.5)

        batch_size = 5
        num_layers = 2
        num_wavelengths = evaluator.wavelengths.shape[0]

        # Create random batch of cell designs
        absorption = ms.Tensor(np.random.rand(batch_size, num_layers, num_wavelengths) * 0.8, ms.float32)
        e_qd_np = np.random.rand(batch_size, num_layers) * 1.5 + 1.0
        v_oc = ms.Tensor(_generic_voc(e_qd_np), ms.float32)
        eqe = ms.Tensor([1.0, 1.0], ms.float32)

        fitness, efficiency, cmi, v_oc_total, j_sc_limit = evaluator(absorption, v_oc, eqe, evaluator.wavelengths, ff=0.75)

        assert fitness.shape == (batch_size,), f"Expected fitness shape ({batch_size},), got {fitness.shape}"
        assert efficiency.shape == (batch_size,), f"Expected efficiency shape ({batch_size},), got {efficiency.shape}"
        assert cmi.shape == (batch_size,), f"Expected cmi shape ({batch_size},), got {cmi.shape}"

        fitness_np = fitness.asnumpy()
        efficiency_np = efficiency.asnumpy()
        cmi_np = cmi.asnumpy()

        assert not np.isnan(fitness_np).any(), "No fitness values should be NaN"
        assert not np.isinf(fitness_np).any(), "No fitness values should be Inf"
        assert np.all(fitness_np > -200), f"Fitness values too negative: {fitness_np}"
        assert np.all(cmi_np >= 0), "CMI values should be non-negative"
        # Penalized fitness always <= raw efficiency
        assert np.all(fitness_np <= efficiency_np + 1e-5), "fitness must not exceed efficiency"

        print(f"Batch processing: {batch_size} designs, fitness=[{fitness_np.min():.4f}, {fitness_np.max():.4f}], cmi=[{cmi_np.min():.4f}, {cmi_np.max():.4f}]")

    def test_bandgap_effect_on_voltage(self):
        """Validates that higher bandgap produces higher voltage."""
        evaluator = SolarPerformanceEvaluator(kappa=0.0)  # No mismatch penalty

        num_wavelengths = evaluator.wavelengths.shape[0]
        absorption = ms.Tensor(np.ones((2, 1, num_wavelengths)) * 0.8, ms.float32)
        eqe = ms.Tensor([1.0], ms.float32)

        # Low vs high bandgap
        e_qd = np.array([[1.2], [1.8]])
        v_oc = ms.Tensor(_generic_voc(e_qd), ms.float32)

        fitness, efficiency, cmi, v_oc_total, j_sc_limit = evaluator(absorption, v_oc, eqe, evaluator.wavelengths, ff=0.75)
        fitness_np = fitness.asnumpy()
        efficiency_np = efficiency.asnumpy()

        assert len(fitness_np) == 2, "Should return 2 fitness values"
        # With kappa=0, fitness == efficiency
        assert np.allclose(fitness_np, efficiency_np, atol=1e-5), \
            "With kappa=0, fitness should equal efficiency"

        print(f"Bandgap effect: E_g=1.2eV -> fitness={fitness_np[0]:.4f}, E_g=1.8eV -> fitness={fitness_np[1]:.4f}")
