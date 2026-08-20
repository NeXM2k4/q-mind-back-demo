import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "materials": ["CdSe", "PbS"],
    "temperature_c": 25.0,
    "spectral_min_nm": 300.0,
    "spectral_max_nm": 1400.0,
}


class TestSimulationEndpoint:
    """Tests for the public POST /api/v1/simulate demo endpoint."""

    def test_run_simulation_tandem(self):
        response = client.post("/api/v1/simulate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert data["efficiency_pct"] > 0
        assert data["voltage_v"] > 0
        assert data["current_ma_cm2"] >= 0
        assert data["computation_time_ms"] >= 0
        assert len(data["layers"]) == 2
        for layer in data["layers"]:
            assert layer["material"] in VALID_PAYLOAD["materials"]
            assert layer["bandgap_ev"] > 0
            assert layer["thickness_nm"] == 300.0
            assert len(layer["absorption_m_inv"]) == len(data["spectrum"]["wavelengths_nm"])

    def test_no_auth_required(self):
        response = client.post("/api/v1/simulate", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_perovskite_and_cdse(self):
        payload = {**VALID_PAYLOAD, "materials": ["Perovskita yoduro bromuro", "CdSe"]}
        response = client.post("/api/v1/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()
        labels = {layer["material"]: layer for layer in data["layers"]}
        assert labels["Perovskita yoduro bromuro"]["control_unit"] == ""
        assert labels["CdSe"]["control_unit"] == "nm"

    def test_perovskite_and_pbs(self):
        payload = {**VALID_PAYLOAD, "materials": ["Perovskita yoduro bromuro", "PbS"]}
        response = client.post("/api/v1/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["fill_factor"] == pytest.approx(0.75, abs=1e-6)

    def test_perovskite_composition_within_stability_bound(self):
        payload = {**VALID_PAYLOAD, "materials": ["Perovskita yoduro bromuro", "PbS"]}
        response = client.post("/api/v1/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()
        perovskite_layer = next(l for l in data["layers"] if l["material"] == "Perovskita yoduro bromuro")
        assert 0.0 <= perovskite_layer["control_value"] <= 0.20

    def test_pbs_diameter_within_validated_range(self):
        payload = {**VALID_PAYLOAD, "materials": ["CdSe", "PbS"]}
        response = client.post("/api/v1/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()
        pbs_layer = next(l for l in data["layers"] if l["material"] == "PbS")
        assert 3.9 <= pbs_layer["control_value"] <= 13.3

    def test_same_material_twice_rejected(self):
        payload = {**VALID_PAYLOAD, "materials": ["CdSe", "CdSe"]}
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 422

    def test_unknown_material_rejected(self):
        payload = {**VALID_PAYLOAD, "materials": ["CdSe", "Silicon"]}
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 422

    def test_single_material_rejected(self):
        payload = {**VALID_PAYLOAD, "materials": ["CdSe"]}
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 422

    def test_inverted_spectral_range_rejected(self):
        payload = {**VALID_PAYLOAD, "spectral_min_nm": 1000.0, "spectral_max_nm": 500.0}
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 422

    def test_temperature_out_of_bounds_rejected(self):
        payload = {**VALID_PAYLOAD, "temperature_c": 100.0}
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 422

    def test_spectrum_matches_requested_window(self):
        payload = {**VALID_PAYLOAD, "spectral_min_nm": 400.0, "spectral_max_nm": 800.0}
        response = client.post("/api/v1/simulate", json=payload)

        assert response.status_code == 200
        data = response.json()
        wavelengths = data["spectrum"]["wavelengths_nm"]
        assert wavelengths[0] == pytest.approx(400.0, abs=0.01)
        assert wavelengths[-1] < 800.0
        assert len(data["spectrum"]["irradiance_w_m2_nm"]) == len(wavelengths)
        assert all(v >= 0 for v in data["spectrum"]["irradiance_w_m2_nm"])

    def test_convergence_history_length(self):
        response = client.post("/api/v1/simulate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        conv = data["convergence"]
        assert len(conv["pce_history"]) == conv["total_generations"]
        assert 1 <= conv["generations_to_convergence"] <= conv["total_generations"]
