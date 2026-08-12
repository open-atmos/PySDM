# pylint: disable=missing-module-docstring
from matplotlib import pyplot
import pytest
import numpy as np
from PySDM.dynamics import SedimentationRemoval0D
from PySDM.physics import si
from PySDM.environments import Box
from PySDM import Builder
from PySDM.backends import ThrustRTC
from PySDM.products import ParticleConcentration, SuperDropletCountPerGridbox, Time


class TestSedimentationRemoval0D:  # pylint: disable=too-few-public-methods,too-many-branches,too-many-locals
    @staticmethod
    @pytest.mark.parametrize("stochastic", (True, False))
    def test_convergence_wrt_dt(backend_class, stochastic, plot=False):
        # arrange
        dts = 0.5 * si.s, 5 * si.s
        dvs = 1e2 * si.m**3, 1e3 * si.m**3, 1e4 * si.m**3
        t_max = 600 * si.s
        multiplicities = 1e5, 1e6, 1e7, 1e8
        water_masses = 1 * si.ug, 2 * si.ug, 3 * si.ug, 4 * si.ug

        if backend_class is ThrustRTC:
            pytest.skip("TODO #1871")

        # act
        output = {}
        for dt in dts:
            for dv in dvs:
                builder = Builder(
                    n_sd=len(multiplicities),
                    environment=Box(dv=dv, dt=dt),
                    backend=backend_class(),
                    dynamics=(
                        SedimentationRemoval0D(
                            stochastic_sedimentation_removal=stochastic
                        ),
                    ),
                )
                particulator = builder.build(
                    attributes={
                        "multiplicity": np.asarray(multiplicities),
                        "signed water mass": np.asarray(water_masses),
                    },
                    products=(
                        ParticleConcentration(),
                        SuperDropletCountPerGridbox(),
                        Time(),
                    ),
                )
                key = f"{dt=} {dv=}"
                output[key] = {name: [] for name in particulator.products}
                while particulator.n_steps * dt <= t_max:
                    if len(output[key]["time"]) != 0:
                        particulator.run(steps=1)
                    for name, product in particulator.products.items():
                        output[key][name].append(product.get() + 0)

        # plot
        pyplot.title(f"{stochastic=}")
        pyplot.xlabel("time [s]")
        pyplot.ylabel("particle concentration [m$^{-3}$]")
        for dt in dts:
            for dv in dvs:
                key = f"{dt=} {dv=}"
                pyplot.plot(
                    output[key]["time"],
                    output[key]["particle concentration"],
                    label=key,
                    linewidth=4 + 3 * np.log10(dt),
                )
        pyplot.gca().set_yscale("log")
        pyplot.gca().set_xlim(left=0, right=t_max)
        pyplot.legend()
        pyplot.grid()

        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        for dt in dts:
            for dv in dvs:
                key = f"{dt=} {dv=}"
                particle_concentration = np.asarray(
                    output[key]["particle concentration"]
                )
                assert particle_concentration[-1] == 0.0

        for dv in dvs:
            time_compare = 0
            for dt in dts:
                key = f"{dt=} {dv=}"
                time_end = np.asarray(output[key]["time"])[-1]
                assert time_end >= time_compare
                time_compare = time_end

        for dt in dts:
            time_compare = 0
            for dv in dvs:
                key = f"{dt=} {dv=}"
                time_end = np.asarray(output[key]["time"])[-1]
                assert time_end >= time_compare
                time_compare = time_end
