"""
unit tests for backend isotope-related routines
"""

import numpy as np
import math
import pytest
from PySDM.backends import CPU

from PySDM import Formulae
from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.physics import si
from PySDM.dynamics import Condensation, IsotopicFractionation, AmbientThermodynamics


class TestIsotopeMethods:
    @staticmethod
    @pytest.mark.parametrize("initial_moles_heavy", (0.5, 1.1))
    @pytest.mark.parametrize("initial_molality_air", (0.1, 0.001, 1.01))
    @pytest.mark.parametrize("bolin_number", (-0.5, 0.88, 2.2))
    def test_isotopic_fractionation(
        backend_class, initial_moles_heavy, initial_molality_air, bolin_number
    ):
        """Checks whether isotopic fractionation updates:
        - heavy-isotope moles in droplets
        - heavy-isotope molality in ambient air
        consistently with the implemented exponential Euler scheme.
        """
        # arrange
        if backend_class.__name__ == "ThrustRTC":
            pytest.xfail("bolin_number not yet supported for ThrustRTC")
        backend_instance = backend_class(Formulae())
        arr2storage = backend_instance.Storage.from_ndarray

        multiplicity = 3
        dm_total = 0.2
        signed_water_mass = 1.0
        molar_mass_heavy_molecule = 2.0

        cell_volume = 2.0
        dry_air_density = 1.0

        moles_heavy = arr2storage(np.array([initial_moles_heavy]))
        molality_in_dry_air = arr2storage(np.array([initial_molality_air]))

        mass_ratio_heavy_to_total = (
            initial_moles_heavy * molar_mass_heavy_molecule / signed_water_mass
        )
        dm_heavy = dm_total / bolin_number * mass_ratio_heavy_to_total
        expected_moles_heavy = initial_moles_heavy * np.exp(
            dm_heavy / initial_moles_heavy
        )
        dn_heavy = expected_moles_heavy - initial_moles_heavy
        mass_of_dry_air = dry_air_density * cell_volume
        expected_molality_air = (
            initial_molality_air - dn_heavy * multiplicity / mass_of_dry_air
        )

        # act
        backend_instance.isotopic_fractionation(
            cell_id=arr2storage(np.array([0], dtype=int)),
            cell_volume=cell_volume,
            multiplicity=arr2storage(np.array([multiplicity])),
            dm_total=arr2storage(np.array([dm_total])),
            signed_water_mass=arr2storage(np.array([signed_water_mass])),
            dry_air_density=arr2storage(np.array([dry_air_density])),
            molar_mass_heavy_molecule=molar_mass_heavy_molecule,
            moles_heavy_molecule=moles_heavy,
            bolin_number=arr2storage(np.array([bolin_number])),
            molality_in_dry_air=molality_in_dry_air,
        )

        # assert
        assert expected_molality_air >= 0
        assert np.isclose(
            moles_heavy.to_ndarray()[0],
            expected_moles_heavy,
        )

        assert np.isclose(
            molality_in_dry_air.to_ndarray()[0],
            expected_molality_air,
        )

    @staticmethod
    def test_bolin_number(backend_class):
        # arrange
        if backend_class.__name__ == "ThrustRTC":
            pytest.xfail("bolin_number not yet supported for ThrustRTC")

        backend = backend_class(
            Formulae(
                isotope_diffusivity_ratios="Stewart1975+GrahamsLaw",
                isotope_relaxation_timescale="ZabaEtAl",
                isotope_equilibrium_fractionation_factors="VanHook1968",
            )
        )

        arr2storage = backend.Storage.from_ndarray

        n_sd = 3
        n_cell = 2

        output = arr2storage(np.empty(n_sd))

        cell_id = arr2storage(np.zeros(n_sd, dtype=np.int64))
        relative_humidity = arr2storage(np.full(n_cell, 0.95))
        temperature = arr2storage(np.full(n_cell, 283.15))
        water_vapour_mixing_ratio = arr2storage(np.full(n_cell, 1.2))
        moles_light_molecule = arr2storage(np.array([1e-3, 2e-3, 3e-3]))
        moles_heavy = arr2storage(np.array([2e-6, 4e-6, 6e-6]))
        molality_in_dry_air = arr2storage(np.full(n_cell, 1e-5))

        # act
        backend.bolin_number(
            output=output,
            cell_id=cell_id,
            isotope="2H",  # TODO
            relative_humidity=relative_humidity,
            temperature=temperature,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio,
            moles_light_molecule=moles_light_molecule,
            moles_heavy=moles_heavy,
            molality_in_dry_air=molality_in_dry_air,
        )

        # assert
        result = output.to_ndarray()

        assert result.shape == (n_sd,)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

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
