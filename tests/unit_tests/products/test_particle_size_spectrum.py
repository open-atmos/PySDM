"""
tests the ParticleSizeSpectrum product against per-mass/per-volume and dry-/wet-size choices
"""

import pytest
import numpy as np
from PySDM import Builder
from PySDM.physics import si
from PySDM.environments import Box
from PySDM.products import (
    ParticleSizeSpectrumPerMassOfDryAir,
    ParticleSizeSpectrumPerVolume,
)
from PySDM.products.size_spectral.particle_size_spectrum import ParticleSizeSpectrum


class TestParticleSizeSpectrum:
    @staticmethod
    @pytest.mark.parametrize(
        "product_class, specific, unit, stp",
        (
            (ParticleSizeSpectrumPerVolume, False, "1.0 / meter ** 4", True),
            (ParticleSizeSpectrumPerVolume, False, "1.0 / meter ** 4", False),
            (
                ParticleSizeSpectrumPerMassOfDryAir,
                True,
                "1.0 / kilogram / meter",
                False,
            ),
        ),
    )
    def test_specific_flag(product_class, specific, unit, backend_instance, stp):
        """checks if the reported concentration is correctly normalised per
        volume or mass of air, and per bin size"""
        # arrange
        dv = 666 * si.m**3
        multiplicity = 1000
        rhod = 44 * si.kg / si.m**3
        min_size = 0
        max_size = 1 * si.mm

        n_sd = 1
        box = Box(dt=np.nan, dv=dv)
        builder = Builder(n_sd=n_sd, backend=backend_instance, environment=box)
        sut = product_class(
            radius_bins_edges=(min_size, max_size), **({"stp": stp} if stp else {})
        )
        builder.build(
            products=(sut,),
            attributes={
                "multiplicity": np.ones(n_sd) * multiplicity,
                "water mass": np.ones(n_sd) * si.ug,
            },
        )
        box["rhod"] = rhod

        # act
        actual = sut.get()

        # assert
        assert sut.unit == unit
        assert sut.specific == specific
        np.testing.assert_approx_equal(
            actual=actual,
            desired=multiplicity
            / dv
            / (max_size - min_size)
            / (rhod if specific or stp else 1)
            * (backend_instance.formulae.constants.rho_STP if stp else 1),
            significant=10,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "product_class",
        (ParticleSizeSpectrumPerVolume, ParticleSizeSpectrumPerMassOfDryAir),
    )
    @pytest.mark.parametrize("dry", (True, False))
    def test_dry_flag(product_class, dry, backend_instance):
        """checks if dry or wet size attribute is correctly picked for moment calculation"""
        # arrange
        n_sd = 1
        dv = 666 * si.m**3
        min_size = 0
        max_size = 1 * si.mm
        multiplicity = 100
        rhod = 44 * si.kg / si.m**3
        wet_volume = 1 * si.um**3
        dry_volume = 0.01 * si.um**3

        box = Box(dt=np.nan, dv=dv)
        builder = Builder(n_sd=n_sd, backend=backend_instance, environment=box)
        sut = product_class(radius_bins_edges=(min_size, max_size), dry=dry)
        builder.build(
            products=(sut,),
            attributes={
                "multiplicity": np.ones(n_sd) * multiplicity,
                "volume": np.ones(n_sd) * (np.nan if dry else wet_volume),
                "dry volume": np.ones(n_sd) * (dry_volume if dry else np.nan),
            },
        )
        box["rhod"] = rhod

        # act
        actual = sut.get()

        # assert
        np.testing.assert_approx_equal(
            actual=actual,
            desired=multiplicity
            / dv
            / (max_size - min_size)
            / (rhod if sut.specific else 1),
            significant=10,
        )

    @staticmethod
    def test_stp_flag_incompatible_with_specific():
        with pytest.raises(Exception, match="precludes specific"):
            ParticleSizeSpectrum(
                stp=True, specific=True, name="", unit="", radius_bins_edges=()
            )
