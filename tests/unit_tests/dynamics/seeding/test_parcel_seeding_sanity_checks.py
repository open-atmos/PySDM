""" tests ensuring expected seeding outcome in some parcel-model based cases """

import numpy as np

from matplotlib import pyplot

from PySDM import Builder, Formulae, products
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation, Coalescence, Seeding
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.environments import Parcel
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si


class TestParcelSeedingSanityChecks:
    @staticmethod
    def test_zero_injection_rate_same_as_no_seeding(backend_class=CPU, plot=False):
        """test that results of parcel simulation are the same without seeding
        as with a zero injection rate"""
        # arrange
        n_sd_seeding = 100
        n_sd_initial = 100
        t_max = 20 * si.min
        timestep = 15 * si.s
        w_min = 0.025 * si.m / si.s
        w_max = 3 * si.m / si.s
        updraft = (
            lambda t: w_min
            + (w_max - w_min) * np.maximum(0, np.sin(t / t_max * 2 * np.pi)) ** 2
        )

        env = Parcel(
            dt=timestep,
            mass_of_dry_air=666 * si.kg,
            p0=1000 * si.hPa,
            initial_water_vapour_mixing_ratio=666 / 30 * si.g / si.kg,
            T0=300 * si.K,
            w=updraft,
        )

        initial_aerosol_dry_radii = Lognormal(
            norm_factor=200 / si.mg * env.mass_of_dry_air,
            m_mode=75 * si.nm,
            s_geom=1.6,
        )
        initial_aerosol_kappa = 0.5

        # parcel with seeding dynamic, zero injection rate
        builder = Builder(
            backend=backend_class(Formulae(seed=100)),
            n_sd=n_sd_seeding + n_sd_initial,
            environment=env,
        )
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(Coalescence(collision_kernel=Geometric()))
        builder.add_dynamic(
            Seeding(
                super_droplet_injection_rate=lambda time: 0,
                seeded_particle_multiplicity=1,
                seeded_particle_extensive_attributes={
                    "water mass": 0.001 * si.ng,
                    "dry volume": 0.0001 * si.ng,
                    "kappa times dry volume": 0.8 * 0.0001 * si.ng,
                },
            )
        )

        r_dry, n_in_dv = ConstantMultiplicity(initial_aerosol_dry_radii).sample(
            n_sd=n_sd_initial, backend=builder.particulator.backend
        )
        attributes = builder.particulator.environment.init_attributes(
            n_in_dv=n_in_dv, kappa=initial_aerosol_kappa, r_dry=r_dry
        )
        particulator = builder.build(
            attributes={
                k: np.pad(
                    array=v,
                    pad_width=(0, n_sd_seeding),
                    mode="constant",
                    constant_values=np.nan if k == "multiplicity" else 0,
                )
                for k, v in attributes.items()
            },
            products=(
                products.SuperDropletCountPerGridbox(name="sd_count"),
                products.Time(),
            ),
        )
        n_steps = int(t_max // timestep)

        output_zero = {
            "attributes": {"water mass": []},
            "products": {"sd_count": [], "time": []},
        }
        for step in range(n_steps + 1):
            if step != 0:
                particulator.run(steps=1)
            for key in output_zero["attributes"].keys():
                data = particulator.attributes[key].to_ndarray(raw=True)
                data[data == 0] = np.nan
                output_zero["attributes"][key].append(data)
            for key in output_zero["products"].keys():
                output_zero["products"][key].append(
                    float(particulator.products[key].get())
                )
        for out in ("attributes", "products"):
            for key in output_zero[out].keys():
                output_zero[out][key] = np.array(output_zero[out][key])

        # parcel with seeding dynamic, positive injection rate
        def injection_rate(time):
            return 1 if 10 * si.min <= time < 15 * si.min else 0

        builder = Builder(
            backend=backend_class(Formulae(seed=100)),
            n_sd=n_sd_seeding + n_sd_initial,
            environment=env,
        )
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(Coalescence(collision_kernel=Geometric()))
        builder.add_dynamic(
            Seeding(
                super_droplet_injection_rate=injection_rate,
                seeded_particle_multiplicity=1,
                seeded_particle_extensive_attributes={
                    "water mass": 0.001 * si.ng,
                    "dry volume": 0.0001 * si.ng,
                    "kappa times dry volume": 0.8 * 0.0001 * si.ng,
                },
            )
        )

        r_dry, n_in_dv = ConstantMultiplicity(initial_aerosol_dry_radii).sample(
            n_sd=n_sd_initial, backend=builder.particulator.backend
        )
        attributes = builder.particulator.environment.init_attributes(
            n_in_dv=n_in_dv, kappa=initial_aerosol_kappa, r_dry=r_dry
        )
        particulator = builder.build(
            attributes={
                k: np.pad(
                    array=v,
                    pad_width=(0, n_sd_seeding),
                    mode="constant",
                    constant_values=np.nan if k == "multiplicity" else 0,
                )
                for k, v in attributes.items()
            },
            products=(
                products.SuperDropletCountPerGridbox(name="sd_count"),
                products.Time(),
            ),
        )
        n_steps = int(t_max // timestep)

        output_pos = {
            "attributes": {"water mass": []},
            "products": {"sd_count": [], "time": []},
        }
        for step in range(n_steps + 1):
            if step != 0:
                particulator.run(steps=1)
            for key in output_pos["attributes"].keys():
                data = particulator.attributes[key].to_ndarray(raw=True)
                data[data == 0] = np.nan
                output_pos["attributes"][key].append(data)
            for key in output_pos["products"].keys():
                output_pos["products"][key].append(
                    float(particulator.products[key].get())
                )
        for out in ("attributes", "products"):
            for key in output_pos[out].keys():
                output_pos[out][key] = np.array(output_pos[out][key])

        # parcel with no seeding dynamic
        builder = Builder(
            backend=backend_class(Formulae(seed=100)),
            n_sd=n_sd_initial,
            environment=env,
        )
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(Coalescence(collision_kernel=Geometric()))

        r_dry, n_in_dv = ConstantMultiplicity(initial_aerosol_dry_radii).sample(
            n_sd=n_sd_initial, backend=builder.particulator.backend
        )
        attributes = builder.particulator.environment.init_attributes(
            n_in_dv=n_in_dv, kappa=initial_aerosol_kappa, r_dry=r_dry
        )

        particulator = builder.build(
            attributes=attributes,
            products=(
                products.SuperDropletCountPerGridbox(name="sd_count"),
                products.Time(),
            ),
        )
        n_steps = int(t_max // timestep)

        output_no_seeding = {
            "attributes": {"water mass": []},
            "products": {"sd_count": [], "time": []},
        }
        for step in range(n_steps + 1):
            if step != 0:
                particulator.run(steps=1)
            for key in output_no_seeding["attributes"].keys():
                data = particulator.attributes[key].to_ndarray(raw=True)
                data[data == 0] = np.nan
                output_no_seeding["attributes"][key].append(data)
            for key in output_no_seeding["products"].keys():
                output_no_seeding["products"][key].append(
                    float(particulator.products[key].get())
                )
        for out in ("attributes", "products"):
            for key in output_no_seeding[out].keys():
                output_no_seeding[out][key] = np.array(output_no_seeding[out][key])

        # plot
        fig, ax = pyplot.subplots(1, 1)
        ax.plot(
            output_zero["products"]["sd_count"],
            output_zero["products"]["time"],
            lw=2,
            label="with seeding dynamic, zero injection rate",
        )
        ax.plot(
            output_pos["products"]["sd_count"],
            output_pos["products"]["time"],
            lw=2,
            ls="--",
            label="with seeding dynamic, positive injection rate",
        )
        ax.plot(
            output_no_seeding["products"]["sd_count"],
            output_no_seeding["products"]["time"],
            lw=2,
            ls=":",
            label="without seeding dynamic",
        )
        ax.set_xlabel("sd_count")
        ax.set_ylabel("time")
        ax.grid()
        ax.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (
            output_zero["products"]["sd_count"]
            == output_no_seeding["products"]["sd_count"]
        ).all()
        assert (np.diff(output_no_seeding["products"]["sd_count"]) == 0).all()
        assert (np.diff(output_zero["products"]["sd_count"]) == 0).all()
        assert (np.diff(output_pos["products"]["sd_count"]) >= 0).all()
