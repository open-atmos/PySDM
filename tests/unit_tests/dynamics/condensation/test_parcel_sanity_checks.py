""" tests ensuring proper condensation solver operation in some parcel-model based cases """

import pytest
from matplotlib import pyplot
from scipy import signal

from PySDM import Builder, Formulae, products
from PySDM.backends import CPU, GPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import discretise_multiplicities, equilibrate_wet_radii
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si


class TestParcelSanityChecks:
    @staticmethod
    @pytest.mark.parametrize(
        "backend_class",
        (
            CPU,
            pytest.param(
                GPU, marks=pytest.mark.xfail(strict=True)
            ),  # TODO #1117 (works with CUDA!)
        ),
    )
    def test_noisy_supersaturation_profiles(
        backend_class, plot=False
    ):  # pylint: disable=too-many-locals
        """cases found using the README parcel snippet"""
        # arrange
        env = Parcel(
            dt=0.25 * si.s,
            mass_of_dry_air=1e3 * si.kg,
            p0=1122 * si.hPa,
            initial_water_vapour_mixing_ratio=20 * si.g / si.kg,
            T0=300 * si.K,
            w=2.5 * si.m / si.s,
        )
        spectrum = Lognormal(norm_factor=1e4 / si.mg, m_mode=50 * si.nm, s_geom=1.5)
        kappa = 0.5 * si.dimensionless
        cloud_range = (0.5 * si.um, 25 * si.um)
        output_interval = 10
        output_points = 200
        n_sd = 64

        formulae = Formulae()
        builder = Builder(backend=backend_class(formulae), n_sd=n_sd, environment=env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        r_dry, specific_concentration = spectral_sampling.Logarithmic(spectrum).sample(
            n_sd
        )
        v_dry = formulae.trivia.volume(radius=r_dry)
        r_wet = equilibrate_wet_radii(
            r_dry=r_dry, environment=env, kappa_times_dry_volume=kappa * v_dry
        )

        attributes = {
            "multiplicity": discretise_multiplicities(
                specific_concentration * env.mass_of_dry_air
            ),
            "dry volume": v_dry,
            "kappa times dry volume": kappa * v_dry,
            "volume": formulae.trivia.volume(radius=r_wet),
        }

        particulator = builder.build(
            attributes,
            products=(
                products.PeakSupersaturation(name="S_max", unit="%"),
                products.EffectiveRadius(
                    name="r_eff", unit="um", radius_range=cloud_range
                ),
                products.ParticleConcentration(
                    name="n_c_cm3", unit="cm^-3", radius_range=cloud_range
                ),
                products.WaterMixingRatio(
                    name="liquid water mixing ratio",
                    unit="g/kg",
                    radius_range=cloud_range,
                ),
                products.ParcelDisplacement(name="z"),
            ),
        )

        cell_id = 0
        output = {
            product.name: [product.get()[cell_id]]
            for product in particulator.products.values()
        }

        for _ in range(output_points):
            particulator.run(steps=output_interval)
            for product in particulator.products.values():
                output[product.name].append(product.get()[cell_id])

        # plot
        fig, axs = pyplot.subplots(1, len(particulator.products) - 1, sharey="all")
        for i, (key, product) in enumerate(particulator.products.items()):
            if key != "z":
                axs[i].plot(output[key], output["z"], marker=".")
                axs[i].set_title(product.name)
                axs[i].set_xlabel(product.unit)
                axs[i].grid()
        axs[0].set_ylabel(particulator.products["z"].unit)
        fig.suptitle(backend_class.__name__)
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        supersaturation_peaks, _ = signal.find_peaks(
            output["S_max"], width=2, prominence=0.01
        )
        assert len(supersaturation_peaks) == 1

    @staticmethod
    @pytest.mark.parametrize("update_thd", (True, False))
    @pytest.mark.parametrize("substeps", (1, 2))
    def test_how_condensation_modifies_args(backend_class, update_thd, substeps):
        """asserting that condensation modifies env thd and water_vapour_mixing_ratio only,
        not rhod"""
        # arrange
        env = Parcel(
            dt=1 * si.s,
            mass_of_dry_air=1 * si.mg,
            p0=1000 * si.hPa,
            initial_water_vapour_mixing_ratio=22.2 * si.g / si.kg,
            T0=300 * si.K,
            w=1 * si.m / si.s,
        )
        builder = Builder(n_sd=10, backend=backend_class(), environment=env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(
            Condensation(
                update_thd=update_thd,
                substeps=substeps,
                adaptive=False,
            )
        )
        particulator = builder.build(
            products=(),
            attributes=builder.particulator.environment.init_attributes(
                kappa=1, r_dry=0.25 * si.um, n_in_dv=1000
            ),
        )

        particulator.dynamics["AmbientThermodynamics"]()

        cell_id = 0
        env = particulator.environment
        pred_rhod_old = env.get_predicted("rhod")[cell_id]
        pred_thd_old = env.get_predicted("thd")[cell_id]
        pred_water_vapour_mixing_ratio_old = env.get_predicted(
            "water_vapour_mixing_ratio"
        )[cell_id]

        env_rhod_old = env["rhod"][cell_id]
        env_thd_old = env["thd"][cell_id]
        env_water_vapour_mixing_ratio_old = env["water_vapour_mixing_ratio"][cell_id]

        # act
        particulator.dynamics["Condensation"]()

        # assert
        assert env_rhod_old == env["rhod"]
        assert env_water_vapour_mixing_ratio_old == env["water_vapour_mixing_ratio"]
        assert env_thd_old == env["thd"]

        assert pred_rhod_old == env.get_predicted("rhod")[cell_id]
        assert (
            pred_water_vapour_mixing_ratio_old
            != env.get_predicted("water_vapour_mixing_ratio")[cell_id]
        )

        if update_thd:
            assert pred_thd_old != env.get_predicted("thd")[cell_id]
        else:
            assert pred_thd_old == env.get_predicted("thd")[cell_id]
