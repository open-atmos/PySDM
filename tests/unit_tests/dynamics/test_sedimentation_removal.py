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
    @pytest.mark.parametrize("all_or_nothing", (True, False))
    def test_convergence_wrt_dt(all_or_nothing, plot=False):
        # arrange
        dt = 1 * si.s
        dv = 666 * si.m**3
        n_steps = 100
        multiplicity = [1, 10, 100, 1000]
        water_mass = [1 * si.ug, 2 * si.ug, 3 * si.ug, 4 * si.ug]
        backend_instance = CPU()

        builder = Builder(
            n_sd=len(multiplicity),
            environment=Box(dv=dv, dt=dt),
            backend=backend_instance,
        )
        builder.add_dynamic(SedimentationRemoval(all_or_nothing=all_or_nothing))
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(multiplicity),
                "signed water mass": np.asarray(water_mass),
            },
            products=(ParticleConcentration(), SuperDropletCountPerGridbox(), Time()),
        )

        # act
        output = {name: [] for name in particulator.products}
        for step in range(n_steps):
            if step != 0:
                particulator.run(steps=1)
            for name, product in particulator.products.items():
                output[name].append(product.get() + 0)

        # plot
        pyplot.title(f"{all_or_nothing=}")
        pyplot.xlabel("time [s]")
        pyplot.ylabel("particle concentration [m$^{-3}$]")
        pyplot.semilogy(output["time"], output["particle concentration"])

        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        # TODO!
