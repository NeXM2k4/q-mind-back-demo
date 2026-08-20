import numpy as np
import mindspore as ms
import pytest
from modelarts_worker.physics.PerovskiteEngine import PerovskiteEngine


def _eg(engine, gene, temperature):
    wavelengths = ms.Tensor(np.array([700.0], dtype=np.float32), ms.float32)
    gene_t = ms.Tensor(np.array([gene], dtype=np.float32), ms.float32)
    temp_t = ms.Tensor(np.array([temperature], dtype=np.float32), ms.float32)
    _, e_g, _ = engine(temperature=temp_t, gene=gene_t, wavelengths=wavelengths)
    return float(e_g.asnumpy().flat[0])


def _voc(engine, gene, temperature):
    wavelengths = ms.Tensor(np.array([700.0], dtype=np.float32), ms.float32)
    gene_t = ms.Tensor(np.array([gene], dtype=np.float32), ms.float32)
    temp_t = ms.Tensor(np.array([temperature], dtype=np.float32), ms.float32)
    _, _, v_oc = engine(temperature=temp_t, gene=gene_t, wavelengths=wavelengths)
    return float(v_oc.asnumpy().flat[0])


def _alpha(engine, wavelength, gene, temperature):
    wavelengths = ms.Tensor(np.array([wavelength], dtype=np.float32), ms.float32)
    gene_t = ms.Tensor(np.array([gene], dtype=np.float32), ms.float32)
    temp_t = ms.Tensor(np.array([temperature], dtype=np.float32), ms.float32)
    absorption, _, _ = engine(temperature=temp_t, gene=gene_t, wavelengths=wavelengths)
    return float(absorption.asnumpy().flat[0])


@pytest.fixture
def engine():
    return PerovskiteEngine()


class TestBandgapAndVoltage:
    """ESPECS.pdf §7.1 — gene, x, E_g, Voc at T=300K."""

    @pytest.mark.parametrize("gene,x,eg_expected,voc_expected", [
        (0.00, 0.000, 1.580000, 1.010000),
        (0.25, 0.050, 1.599325, 1.029325),
        (0.50, 0.100, 1.620300, 1.050300),
        (0.75, 0.150, 1.642925, 1.072925),
        (1.00, 0.200, 1.667200, 1.097200),
    ])
    def test_eg_and_voc_at_300k(self, engine, gene, x, eg_expected, voc_expected):
        eg = _eg(engine, gene, 300.0)
        voc = _voc(engine, gene, 300.0)
        assert eg == pytest.approx(eg_expected, abs=1e-5)
        assert voc == pytest.approx(voc_expected, abs=1e-5)


class TestTemperatureDependence:
    """ESPECS.pdf §7.2 — gene=0.5, varying T."""

    @pytest.mark.parametrize("temperature,eg_expected", [
        (273.15, 1.612245),
        (298.15, 1.619745),
        (300.00, 1.620300),
        (325.00, 1.627800),
        (350.00, 1.635300),
    ])
    def test_eg_at_temperature(self, engine, temperature, eg_expected):
        eg = _eg(engine, 0.5, temperature)
        assert eg == pytest.approx(eg_expected, abs=1e-5)


class TestAbsorptionProfile:
    """ESPECS.pdf §7.3 — gene=0.5, T=300K, E_g=1.620300 eV."""

    @pytest.mark.parametrize("wavelength,alpha_expected,rel_tol", [
        (400.00, 7.0000e6, 1e-3),
        (600.00, 7.0000e6, 1e-3),
        (720.00, 7.3742e6, 1e-3),
        (765.19, 6.5000e6, 2e-3),
        (800.00, 1.1554e6, 1e-3),
        (900.00, 2.3161e1, 5e-2),
    ])
    def test_alpha_at_wavelength(self, engine, wavelength, alpha_expected, rel_tol):
        alpha = _alpha(engine, wavelength, 0.5, 300.0)
        assert alpha == pytest.approx(alpha_expected, rel=rel_tol)


class TestInvariants:
    """ESPECS.pdf §8 — automated invariants."""

    def test_dominio(self, engine):
        assert _eg(engine, 0.0, 300.0) == pytest.approx(1.580000, abs=1e-5)
        assert _eg(engine, 1.0, 300.0) == pytest.approx(1.667200, abs=1e-5)

    def test_monotonicidad(self, engine):
        genes = np.linspace(0, 1, 100)
        egs = [_eg(engine, float(g), 300.0) for g in genes]
        assert all(np.diff(egs) > 0)

    def test_estabilidad(self, engine):
        x_at_gene_1 = engine.X_MIN + 1.0 * (engine.X_MAX - engine.X_MIN)
        assert x_at_gene_1 <= 0.20

    def test_signo_termico(self, engine):
        assert _eg(engine, 0.5, 350.0) > _eg(engine, 0.5, 300.0)

    def test_voc_positivo(self, engine):
        genes = np.linspace(0, 1, 100)
        assert all(_voc(engine, float(g), 300.0) > 0 for g in genes)

    def test_absorcion_sobre_gap(self, engine):
        alpha = _alpha(engine, 400.0, 0.5, 300.0)
        assert alpha == pytest.approx(engine.ALPHA_MAX * engine.W_CONT, rel=1e-3)

    def test_absorcion_bajo_gap(self, engine):
        alpha = _alpha(engine, 900.0, 0.5, 300.0)
        assert alpha < engine.ALPHA_MAX * 1e-4
