from PySDM.dynamics import SedimentationRemoval
from PySDM.physics import si
from PySDM.environments import Box
from PySDM.builder import Builder
from PySDM.backends import CPU
from PySDM.products import ParticleConcentration, SuperDropletCountPerGridbox, Time
from matplotlib import pyplot
import pytest
import numpy as np


class TestSedimentationRemoval:
    @staticmethod
    @pytest.mark.parametrize("all_or_nothing", (False,))
    def test_convergence_wrt_dt(all_or_nothing, plot=True):
        # arrange
        dts = 5 * si.s, 0.5 * si.s
        dvs = 1e2 * si.m**3, 1e3 * si.m**3, 1e4 * si.m**3
        t_max = 500 * si.s
        multiplicities = 1e5, 1e6, 1e7, 1e8
        water_masses = 1 * si.ug, 2 * si.ug, 3 * si.ug, 4 * si.ug
        backend_instance = CPU()

        # act
        output = {}
        for dt in dts:
            for dv in dvs:
                builder = Builder(
                    n_sd=len(multiplicities),
                    environment=Box(dv=dv, dt=dt),
                    backend=backend_instance,
                )
                builder.add_dynamic(SedimentationRemoval(all_or_nothing=all_or_nothing))
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
        pyplot.title(f"{all_or_nothing=}")
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
        # TODO!
