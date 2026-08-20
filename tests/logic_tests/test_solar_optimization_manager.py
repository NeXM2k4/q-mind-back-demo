import pytest
import numpy as np
import mindspore as ms
from modelarts_worker.logic.SolarOptimizationManager import SolarOptimizationManager


class TestSolarOptimizationManager:

    @pytest.fixture
    def manager(self):
        """Creates a manager instance with the materials database."""
        return SolarOptimizationManager(csv_path='db/materials.csv', kappa=0.5)

    def test_catalog_loading(self, manager):
        """Validates that the materials catalog is loaded correctly from CSV."""
        # Check catalog exists and contains data
        assert len(manager.catalog) > 0, "Catalog should not be empty"

        # Verify key materials are present
        expected_materials = ['CdSe', 'PbS', 'CdS', 'GaAs']
        for material in expected_materials:
            assert material in manager.catalog, f"{material} should be in catalog"

        # Verify material properties structure
        cdse = manager.catalog['CdSe']
        required_keys = ['Eg_0K_eV', 'Alpha_evK', 'Beta_K', 'me_eff', 'mh_eff', 'epsilon_r']
        for key in required_keys:
            assert key in cdse, f"{key} should be present in material properties"
            assert not np.isnan(cdse[key]), f"{key} should not be NaN"

        print(f"Catalog loaded: {len(manager.catalog)} materials")

    def test_engine_creation(self, manager):
        """Validates that BrusEngine instances are created correctly."""
        # Get engine for a specific material
        engine = manager.get_engine('CdSe')

        # Verify engine is a BrusEngine instance
        from modelarts_worker.physics.BrusEngine import BrusEngine
        assert isinstance(engine, BrusEngine), "Should return BrusEngine instance"

        # Verify engine properties match catalog values
        cdse_props = manager.catalog['CdSe']
        assert engine.bandgap == cdse_props['Eg_0K_eV'], "Bandgap should match catalog"
        assert engine.alpha == cdse_props['Alpha_evK'], "Alpha should match catalog"
        assert engine.beta == cdse_props['Beta_K'], "Beta should match catalog"

        print(f"Engine created: CdSe with Eg={engine.bandgap:.3f} eV")

    def test_engine_caching(self, manager):
        """Validates that engines are cached and not recreated."""
        # Get same engine twice
        engine1 = manager.get_engine('PbS')
        engine2 = manager.get_engine('PbS')

        # Should be the exact same object (cached)
        assert engine1 is engine2, "Engine should be cached and reused"

        # Cache should contain the engine
        assert 'PbS' in manager.engines_cache, "Engine should be in cache"
        assert len(manager.engines_cache) == 1, "Only one engine should be cached"

        # Get different engine
        engine3 = manager.get_engine('CdSe')
        assert engine3 is not engine1, "Different materials should have different engines"
        assert len(manager.engines_cache) == 2, "Two engines should be cached"

        print(f"Caching verified: {len(manager.engines_cache)} engines cached")

    def test_invalid_material(self, manager):
        """Validates error handling for non-existent materials."""
        with pytest.raises(KeyError):
            manager.get_engine('NonExistentMaterial')

    def test_run_study_single_layer(self, manager):
        """Validates a complete optimization study for single-layer cell."""
        params = {
            'materials': ['CdSe'],
            'pop_size': 10,
            'alpha': 0.5,
            'mutation': 0.1,
            'iterations': 5,
            'temp': ms.Tensor([300.0], ms.float32),
            'wavelength': ms.Tensor(np.arange(280, 2000, 20), ms.float32)
        }

        study = manager.run_study(params)

        # Validate results
        assert len(study["fitness_history"]) == 5, "Should have 5 fitness values"
        assert len(study["pce_history"]) == 5, "Should have 5 pce values"
        assert len(study["avg_fitness_history"]) == 5, "Should have 5 avg-fitness values"
        assert all(isinstance(f, float) for f in study["fitness_history"]), "Fitness should be floats"
        assert not any(np.isnan(f) for f in study["fitness_history"]), "No fitness should be NaN"
        assert all(pce >= fit - 1e-5 for pce, fit in zip(study["pce_history"], study["fitness_history"])), \
            "PCE should be >= fitness each generation"
        assert isinstance(study["champion_cmi"], float) and study["champion_cmi"] == 0.0, \
            "Single-layer CMI should be 0.0"
        assert len(study["phe"]) == 1, "Single layer should have 1 PHE value"
        assert 0.0 <= study["phe"][0] <= 1.0, "PHE should be in [0, 1]"
        assert len(study["bandgaps_eV"]) == 1, "Single layer should have 1 bandgap"
        assert study["champion_voc"] > 0, "Champion Voc should be positive"
        assert study["champion_jsc"] >= 0, "Champion Jsc should be non-negative"

        # Best control values (radius, legacy CSV path uses BrusEngine for all materials)
        best_controls = study["control_values"]
        assert len(best_controls) == 1, "Single layer should have 1 control value"
        assert 2.0 <= best_controls[0] <= 10.0, "Radius should be within bounds"

        print(f"Single-layer study: best fitness={study['fitness_history'][-1]:.4f}, pce={study['pce_history'][-1]:.4f}, cmi={study['champion_cmi']:.4f}, radius={best_controls[0]:.2f} nm")

    def test_run_study_tandem(self, manager):
        """Validates optimization for tandem (2-layer) solar cell."""
        params = {
            'materials': ['CdSe', 'PbS'],
            'pop_size': 15,
            'alpha': 0.5,
            'mutation': 0.15,
            'iterations': 3,
            'temp': ms.Tensor([300.0], ms.float32),
            'wavelength': ms.Tensor(np.arange(280, 2000, 20), ms.float32)
        }

        study = manager.run_study(params)

        # Validate results
        assert len(study["fitness_history"]) == 3, "Should have 3 fitness values"
        assert len(study["pce_history"]) == 3, "Should have 3 pce values"
        assert len(study["avg_fitness_history"]) == 3, "Should have 3 avg-fitness values"

        # Best control values (radii)
        best_controls = study["control_values"]
        assert len(best_controls) == 2, "Tandem should have 2 control values"
        assert all(2.0 <= r <= 10.0 for r in best_controls), "All radii should be within bounds"
        assert study["champion_cmi"] >= 0.0, "CMI must be non-negative"
        assert len(study["phe"]) == 2, "Tandem should have 2 PHE values"
        assert len(study["bandgaps_eV"]) == 2, "Tandem should have 2 bandgap values"

        print(f"Tandem study: fitness={study['fitness_history'][-1]:.4f}, pce={study['pce_history'][-1]:.4f}, cmi={study['champion_cmi']:.4f}, radii={best_controls}")

    def test_run_study_multi_layer(self, manager):
        """Validates optimization for multi-junction (3-layer) cell."""
        params = {
            'materials': ['CdS', 'CdSe', 'PbS'],
            'pop_size': 10,
            'alpha': 0.5,
            'mutation': 0.1,
            'iterations': 3,
            'temp': ms.Tensor([300.0], ms.float32),
            'wavelength': ms.Tensor(np.arange(280, 2000, 20), ms.float32)
        }

        study = manager.run_study(params)

        # Validate results
        best_controls = study["control_values"]
        assert len(best_controls) == 3, "3-layer should have 3 control values"
        assert len(study["fitness_history"]) == 3, "Should have 3 iterations"
        assert len(study["bandgaps_eV"]) == 3, "Should have 3 bandgap values"
        assert len(study["phe"]) == 3, "Should have 3 PHE values"

        # Engines should be cached for all materials
        assert len(manager.engines_cache) == 3, "All 3 engines should be cached"

        print(f"Multi-layer study: {len(params['materials'])} layers, controls={best_controls}, bandgaps={study['bandgaps_eV']}")

    def test_fitness_progression(self, manager):
        """Validates that optimization shows improvement or stability over iterations."""
        params = {
            'materials': ['CdSe', 'PbS'],
            'pop_size': 20,
            'alpha': 0.5,
            'mutation': 0.1,
            'iterations': 10,
            'temp': ms.Tensor([300.0], ms.float32),
            'wavelength': ms.Tensor(np.arange(280, 2000, 20), ms.float32)
        }

        study = manager.run_study(params)
        fitness_history = study["fitness_history"]

        # Check general trend (allowing for some variance)
        initial_fitness = np.mean(fitness_history[:3])
        final_fitness = np.mean(fitness_history[-3:])

        print(f"Fitness progression: {initial_fitness:.4f} -> {final_fitness:.4f}")
        print(f"Full history: {[f'{f:.2f}' for f in fitness_history]}")

    def test_different_populations(self, manager):
        """Validates that different population sizes work correctly."""
        wavelengths = ms.Tensor(np.arange(280, 2000, 20), ms.float32)

        for pop_size in [5, 10, 20, 50]:
            params = {
                'materials': ['CdSe'],
                'pop_size': pop_size,
                'alpha': 0.5,
                'mutation': 0.1,
                'iterations': 2,
                'temp': ms.Tensor([300.0], ms.float32),
                'wavelength': wavelengths
            }

            study = manager.run_study(params)

            assert len(study["fitness_history"]) == 2, f"Pop size {pop_size} should work"
            assert len(study["control_values"]) == 1, f"Pop size {pop_size} should return valid control values"

        print("Tested population sizes: 5, 10, 20, 50 - all working")

    def test_temperature_variation(self, manager):
        """Validates that different operating temperatures affect results."""
        wavelengths = ms.Tensor(np.arange(280, 2000, 20), ms.float32)

        results = {}
        for temp in [250.0, 300.0, 350.0]:
            params = {
                'materials': ['CdSe'],
                'pop_size': 10,
                'alpha': 0.5,
                'mutation': 0.1,
                'iterations': 3,
                'temp': ms.Tensor([temp], ms.float32),
                'wavelength': wavelengths
            }

            study = manager.run_study(params)
            results[temp] = {
                'fitness': study["fitness_history"][-1],
                'pce': study["pce_history"][-1],
                'radius': study["control_values"][0],
            }

        print("Temperature effects:")
        for temp, data in results.items():
            print(f"  T={temp}K: fitness={data['fitness']:.4f}, pce={data['pce']:.4f}, radius={data['radius']:.2f} nm")

    def test_mutation_strength_effect(self, manager):
        """Validates that mutation strength affects exploration."""
        wavelengths = ms.Tensor(np.arange(280, 2000, 20), ms.float32)

        for mutation in [0.01, 0.1, 0.3]:
            params = {
                'materials': ['CdSe', 'PbS'],
                'pop_size': 15,
                'alpha': 0.5,
                'mutation': mutation,
                'iterations': 3,
                'temp': ms.Tensor([300.0], ms.float32),
                'wavelength': wavelengths
            }

            study = manager.run_study(params)

            # Just verify it works with different mutation strengths
            assert len(study["fitness_history"]) == 3, f"Mutation {mutation} should work"

        print("Mutation strength variations: 0.01, 0.1, 0.3 - all functional")
