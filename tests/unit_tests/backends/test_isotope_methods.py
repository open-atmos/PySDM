"""
unit tests for backend isotope-related routines
"""

import numpy as np


class TestIsotopeMethods:
    @staticmethod
    def test_isotopic_fractionation(backend_instance):
        """checks if for a given dm_dt of total liquid water, the changes
        in the amounts of moles of heavy isotopes in droplets,
        and in the ambient air are OK"""
        # arrange
        backend = backend_instance
        arr2storage = backend.Storage.from_ndarray

        moles_heavy = arr2storage(np.array([0.5]))
        molality_in_dry_air = arr2storage(np.array([0.1]))

        expected_moles_heavy = 0.5 + 0.05
        mass_of_dry_air = 1.0 * 2.0
        expected_molality_air = 0.1 - (0.05 * 3.0 / mass_of_dry_air)

        # act
        backend.isotopic_fractionation(
            cell_id=arr2storage(np.array([0], dtype=int)),
            cell_volume=arr2storage(np.array([2.0])),
            multiplicity=arr2storage(np.array([3.0])),
            dm_total=arr2storage(np.array([0.2])),
            signed_water_mass=arr2storage(np.array([1.0])),
            dry_air_density=arr2storage(np.array([1.0])),
            molar_mass_heavy_molecule=2.0,
            moles_heavy_molecule=moles_heavy,
            bolin_number=arr2storage(np.array([2.0])),
            molality_in_dry_air=molality_in_dry_air,
        )

        # assert
        assert np.isclose(moles_heavy.to_ndarray()[0], expected_moles_heavy)
        assert np.isclose(molality_in_dry_air.to_ndarray()[0], expected_molality_air)

    @staticmethod
    def test_isotopic_delta(backend_instance):
        # arrange
        backend = backend_instance
        arr2storage = backend.Storage.from_ndarray
        n_sd = 10
        output = arr2storage(np.empty(n_sd))
        ratio = arr2storage(np.zeros(n_sd))

        # act
        backend.isotopic_delta(output=output, ratio=ratio, reference_ratio=0.0001)

        # assert
        assert (output.to_ndarray() == -1).all()
