import pytest
import pandas as pd
import numpy as np
import mindspore as ms
from modelarts_worker.physics.BrusEngine import BrusEngine

def load_material_data():
    """Load material properties from CSV file."""
    df = pd.read_csv('db/materials.csv')
    return df.to_dict(orient='records')

def _gene_for_radius(engine, radius):
    """Converts a physical radius [nm] to the unit gene the engine now expects."""
    return (radius - engine.r_min) / (engine.r_max - engine.r_min)

class TestBrusEngineMaterials:

    @pytest.mark.parametrize("material_props", load_material_data())
    def test_material_physics(self, material_props):
        """
        Validates that the engine correctly processes each material from the CSV.
        """
        name = material_props['Material']

        engine = BrusEngine(
            bandgap=material_props['Eg_0K_eV'],
            alpha=material_props['Alpha_evK'],
            beta=material_props['Beta_K'],
            me_eff=material_props['me_eff'],
            mh_eff=material_props['mh_eff'],
            eps_r=material_props['epsilon_r']
        )

        temp = ms.Tensor([300.0], ms.float32)
        gene = ms.Tensor([_gene_for_radius(engine, 3.0)], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)

        absorption, e_qd, v_oc = engine(temp, gene, wavelengths)

        # Validate absorption tensor
        assert isinstance(absorption, ms.Tensor), f"Error in {name}: Absorption is not a Tensor"
        abs_np = absorption.asnumpy()
        assert abs_np.shape == (2800,), f"Error in {name}: Unexpected shape {abs_np.shape}"
        assert not np.isnan(abs_np).any(), f"Error in {name}: NaN values in absorption"
        assert not np.isinf(abs_np).any(), f"Error in {name}: Inf values in absorption"
        assert np.all(abs_np >= 0), f"Error in {name}: Negative absorption values"
        assert np.max(abs_np) > 0, f"Error in {name}: Absorption is zero everywhere"

        # Validate quantum dot bandgap
        assert isinstance(e_qd, ms.Tensor), f"Error in {name}: e_qd is not a Tensor"
        e_qd_val = e_qd.asnumpy().item()
        assert not np.isnan(e_qd_val), f"Error in {name}: e_qd is NaN"

        # Quantum confinement effect: e_qd can be larger or smaller than bulk depending on radius and material
        # Validate the change is within physically reasonable bounds
        bandgap_change = e_qd_val - material_props['Eg_0K_eV']
        assert abs(bandgap_change) < 5.0, f"Error in {name}: Bandgap change ({bandgap_change:.3f} eV) is too large"
        assert e_qd_val > 0.1, f"Error in {name}: QD bandgap ({e_qd_val:.3f} eV) is too small"
        assert e_qd_val < 10.0, f"Error in {name}: QD bandgap ({e_qd_val:.3f} eV) is unrealistically high"

        # Validate V_oc
        assert isinstance(v_oc, ms.Tensor), f"Error in {name}: v_oc is not a Tensor"
        v_oc_val = v_oc.asnumpy().item()
        assert not np.isnan(v_oc_val), f"Error in {name}: v_oc is NaN"
        assert v_oc_val >= 0.0, f"Error in {name}: v_oc should be non-negative"

        # Validate the absorption profile: the peak+continuum model saturates just
        # above the gap (unlike a pure Gaussian, the argmax needn't sit exactly at
        # the gap energy), so check the value AT the gap is a substantial fraction
        # of the true maximum, and that it decays well below the gap.
        max_abs = np.max(abs_np)
        gap_wavelength = 1239.84 / e_qd_val
        gap_idx = int(np.clip(round(gap_wavelength) - 200, 0, len(abs_np) - 1))
        assert abs_np[gap_idx] >= max_abs * 0.5,             f"Error in {name}: Absorption at the gap ({abs_np[gap_idx]:.2e}) should be a substantial fraction of the max ({max_abs:.2e})"
        if gap_idx < len(abs_np) - 150:
            far_below_gap_avg = np.mean(abs_np[gap_idx + 100:gap_idx + 150])
            assert far_below_gap_avg < max_abs * 0.3,                 f"Error in {name}: Absorption doesn't decay well below the gap"

        print(f"{name}: e_qd={e_qd_val:.3f} eV, v_oc={v_oc_val:.3f} V, gap={gap_wavelength:.1f} nm, max_abs={np.max(abs_np):.2e}")

    def test_csv_structure(self):
        """Verifies that the CSV file has all required columns."""
        df = pd.read_csv('db/materials.csv')
        required_columns = ['Material', 'Eg_0K_eV', 'Alpha_evK', 'Beta_K', 'me_eff', 'mh_eff', 'epsilon_r']
        for col in required_columns:
            assert col in df.columns, f"Missing critical column: {col}"

    def test_size_dependence(self):
        """Validates that smaller QDs have larger bandgaps (quantum confinement)."""
        engine = BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, me_eff=0.07, mh_eff=0.45, eps_r=10.0)

        temp = ms.Tensor([300.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)

        gene_small = ms.Tensor([_gene_for_radius(engine, 2.0)], ms.float32)
        gene_large = ms.Tensor([_gene_for_radius(engine, 5.0)], ms.float32)

        _, e_qd_small, _ = engine(temp, gene_small, wavelengths)
        _, e_qd_large, _ = engine(temp, gene_large, wavelengths)

        e_small = e_qd_small.asnumpy().item()
        e_large = e_qd_large.asnumpy().item()

        assert e_small > e_large, f"Smaller QD should have larger bandgap: {e_small:.3f} eV vs {e_large:.3f} eV"
        print(f"Quantum confinement: E(2nm)={e_small:.3f} eV > E(5nm)={e_large:.3f} eV")

    def test_temperature_dependence(self):
        """Validates that bandgap decreases with temperature (Varshni's law)."""
        engine = BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, me_eff=0.07, mh_eff=0.45, eps_r=10.0)

        gene = ms.Tensor([_gene_for_radius(engine, 3.0)], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)

        temp_low = ms.Tensor([100.0], ms.float32)
        temp_high = ms.Tensor([500.0], ms.float32)

        _, e_qd_low, _ = engine(temp_low, gene, wavelengths)
        _, e_qd_high, _ = engine(temp_high, gene, wavelengths)

        e_low = e_qd_low.asnumpy().item()
        e_high = e_qd_high.asnumpy().item()

        assert e_low > e_high, f"Bandgap should decrease with temperature: E(100K)={e_low:.3f} eV vs E(500K)={e_high:.3f} eV"
        print(f"Varshni effect: E(100K)={e_low:.3f} eV > E(500K)={e_high:.3f} eV")

    def test_edge_cases(self):
        """Validates behavior at the engine's own radius bounds (gene=0 and gene=1)."""
        engine = BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, me_eff=0.07, mh_eff=0.45, eps_r=10.0)

        temp = ms.Tensor([300.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)

        # gene=0 -> r_min (strong confinement)
        gene_tiny = ms.Tensor([0.0], ms.float32)
        abs_tiny, e_qd_tiny, _ = engine(temp, gene_tiny, wavelengths)
        e_tiny = e_qd_tiny.asnumpy().item()

        assert not np.isnan(abs_tiny.asnumpy()).any(), "NaN values at r_min"
        assert e_tiny > 1.5, f"Strong confinement should increase bandgap: {e_tiny:.3f} eV"

        # gene=1 -> r_max (weak confinement, approaches bulk)
        gene_large = ms.Tensor([1.0], ms.float32)
        abs_large, e_qd_large, _ = engine(temp, gene_large, wavelengths)
        e_large = e_qd_large.asnumpy().item()

        assert not np.isnan(abs_large.asnumpy()).any(), "NaN values at r_max"
        assert abs(e_large - 1.5) < 0.5, f"Large QD should approach bulk bandgap: {e_large:.3f} eV"

        print(f"Edge cases: E(r_min)={e_tiny:.3f} eV, E(r_max)={e_large:.3f} eV")
