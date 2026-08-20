import numpy as np
import mindspore as ms
import pytest
from modelarts_worker.physics.PbSEngine import PbSEngine


def _run(engine, gene, temperature, wavelength=700.0):
    wavelengths = ms.Tensor(np.array([wavelength], dtype=np.float32), ms.float32)
    gene_t = ms.Tensor(np.array([gene], dtype=np.float32), ms.float32)
    temp_t = ms.Tensor(np.array([temperature], dtype=np.float32), ms.float32)
    absorption, e_g, v_oc = engine(temperature=temp_t, gene=gene_t, wavelengths=wavelengths)
    return float(absorption.asnumpy().flat[0]), float(e_g.asnumpy().flat[0]), float(v_oc.asnumpy().flat[0])


@pytest.fixture
def engine():
    return PbSEngine()


class TestMoreelsCompatibility:
    """ESPECS.pdf §10 — tandem compatibility table (informational, not a bound test vector)."""

    @pytest.mark.parametrize("diameter,eg_expected", [(4.16, 1.030), (3.96, 1.070)])
    def test_eg_matches_moreels_at_300k(self, engine, diameter, eg_expected):
        gene = (diameter - engine.D_MIN) / (engine.D_MAX - engine.D_MIN)
        _, e_g, _ = _run(engine, gene, 300.0)
        assert e_g == pytest.approx(eg_expected, abs=2e-3)


class TestDomainAndInvariants:
    def test_domain_bounds(self, engine):
        _, eg_min, _ = _run(engine, 0.0, 300.0)
        _, eg_max, _ = _run(engine, 1.0, 300.0)
        assert eg_min > eg_max, "Larger diameter (gene=1) must have a smaller bandgap"

    def test_monotonic_with_diameter(self, engine):
        genes = np.linspace(0, 1, 50)
        egs = [_run(engine, float(g), 300.0)[1] for g in genes]
        assert all(np.diff(egs) < 0)

    def test_anomalous_positive_thermal_coefficient(self, engine):
        """PbS shares the Hoke-anomalous +dEg/dT behavior with perovskite (ESPECS.pdf §2)."""
        _, eg_low, _ = _run(engine, 0.5, 273.15)
        _, eg_high, _ = _run(engine, 0.5, 350.0)
        assert eg_high > eg_low

    def test_eg_unchanged_at_reference_temperature(self, engine):
        gene = 0.4
        _, eg_at_ref, _ = _run(engine, gene, engine.T_REF)
        d = engine.D_MIN + gene * (engine.D_MAX - engine.D_MIN)
        eg_moreels = engine.MOREELS_EG0 + 1.0 / (engine.MOREELS_A * d**2 + engine.MOREELS_B * d)
        assert eg_at_ref == pytest.approx(eg_moreels, abs=1e-5)

    def test_voc_positive_across_domain(self, engine):
        genes = np.linspace(0, 1, 50)
        assert all(_run(engine, float(g), 300.0)[2] > 0 for g in genes)

    def test_voc_formula(self, engine):
        _, eg, voc = _run(engine, 0.5, 300.0)
        assert voc == pytest.approx(max(engine.VOC_SLOPE * eg - engine.VOC_OFFSET, 0.0), abs=1e-5)

    def test_absorption_saturates_above_gap(self, engine):
        _, eg, _ = _run(engine, 0.5, 300.0)
        alpha_far_above_gap = _run(engine, 0.5, 300.0, wavelength=1239.84 / (eg + 1.0))[0]
        assert alpha_far_above_gap == pytest.approx(engine.alpha_max * engine.W_CONT, rel=1e-3)

    def test_absorption_no_nan(self, engine):
        alpha, eg, voc = _run(engine, 0.3, 310.0, wavelength=850.0)
        assert not np.isnan(alpha)
        assert not np.isnan(eg)
        assert not np.isnan(voc)
        assert alpha >= 0
