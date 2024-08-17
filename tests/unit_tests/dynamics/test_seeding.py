""" a not-so-unit test checking that results of a box simulation
are the same without seeding as with a zero injection rate """

import numpy as np

from matplotlib import pyplot

from PySDM import Builder
from PySDM.products import SuperDropletCountPerGridbox, Time
from PySDM.backends import CPU
from PySDM.dynamics import Seeding
from PySDM.environments import Box
from PySDM.physics import si


def test_zero_injection_rate_same_as_no_seeding(plot=False, backend_instance=CPU()):
    # arrange
    def simulation(*, dynamics):
        n_sd_seeding = 100
        n_sd_initial = 100
        t_max = 20 * si.min
        timestep = 15 * si.s
        dv = 1 * si.cm**3

        builder = Builder(
            backend=backend_instance,
            n_sd=n_sd_seeding + n_sd_initial,
            environment=Box(dt=timestep, dv=dv),
        )
        for dynamic in dynamics:
            builder.add_dynamic(dynamic)

        particulator = builder.build(
            attributes={
                k: np.pad(
                    array=v,
                    pad_width=(0, n_sd_seeding),
                    mode="constant",
                    constant_values=np.nan if k == "multiplicity" else 0,
                )
                for k, v in {
                    "volume": np.ones(n_sd_initial) * si.um**3,
                    "multiplicity": np.ones(n_sd_initial),
                }.items()
            },
            products=(
                SuperDropletCountPerGridbox(name="sd_count"),
                Time(),
            ),
        )
        products = {"sd_count": [], "time": []}
        for step in range(int(t_max // timestep) + 1):
            if step != 0:
                particulator.run(steps=1)
            for key, val in products.items():
                val.append(float(particulator.products[key].get()))
        for key in products:
            products[key] = np.array(products[key])
        return products

    # act
    common_seeding_ctor_args = {
        "seeded_particle_multiplicity": 1,
        "seeded_particle_extensive_attributes": {
            "water mass": 0.001 * si.ng,
        },
    }
    output = {
        "zero_injection_rate": simulation(
            dynamics=(
                Seeding(
                    super_droplet_injection_rate=lambda time: 0,
                    **common_seeding_ctor_args
                ),
            )
        ),
        "positive_injection_rate": simulation(
            dynamics=(
                Seeding(
                    super_droplet_injection_rate=lambda time: (
                        1 if 10 * si.min <= time < 15 * si.min else 0
                    ),
                    **common_seeding_ctor_args
                ),
            )
        ),
        "no_seeding": simulation(dynamics=()),
    }

    # plot
    pyplot.plot(
        output["zero_injection_rate"]["sd_count"],
        output["zero_injection_rate"]["time"],
        lw=2,
        label="with seeding dynamic, zero injection rate",
    )
    pyplot.plot(
        output["positive_injection_rate"]["sd_count"],
        output["positive_injection_rate"]["time"],
        lw=2,
        ls="--",
        label="with seeding dynamic, positive injection rate",
    )
    pyplot.plot(
        output["no_seeding"]["sd_count"],
        output["no_seeding"]["time"],
        lw=2,
        ls=":",
        label="without seeding dynamic",
    )
    pyplot.xlabel("sd_count")
    pyplot.ylabel("time")
    pyplot.grid()
    pyplot.legend()
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    np.testing.assert_array_equal(
        output["zero_injection_rate"]["sd_count"], output["no_seeding"]["sd_count"]
    )
    np.testing.assert_array_equal(np.diff(output["no_seeding"]["sd_count"]), 0)
    np.testing.assert_array_equal(np.diff(output["zero_injection_rate"]["sd_count"]), 0)
    assert np.amax(np.diff(output["positive_injection_rate"]["sd_count"])) >= 0
