import numpy as np

from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.environments import Box
from PySDM.physics import si


def make_one_step_and_record_flag(particulator, output, enable):
    particulator.dynamics["Collision"].enable = enable
    particulator.run(steps=1)
    output += [particulator.attributes["flag coalescence"].to_ndarray().copy()]


def test_flag_coalescence(backend_class, n_sd=1000, golovin_b=1e15):
    # arrange
    builder = Builder(
        backend=backend_class(), n_sd=n_sd, environment=Box(dv=1 * si.m**3, dt=1 * si.s)
    )
    builder.add_dynamic(Coalescence(collision_kernel=Golovin(b=golovin_b)))
    builder.request_attribute("flag coalescence")
    particulator = builder.build(
        attributes={
            "multiplicity": np.full(n_sd, fill_value=1e5),
            "water mass": np.full(n_sd, fill_value=1 * si.ug),
        }
    )

    # act
    output = []
    make_one_step_and_record_flag(particulator, output, enable=False)
    make_one_step_and_record_flag(particulator, output, enable=True)
    make_one_step_and_record_flag(particulator, output, enable=False)
    make_one_step_and_record_flag(particulator, output, enable=True)

    # assert
    assert (output[0] == False).all()
    assert (output[1] == True).any()
    assert (output[2] == False).all()
    assert (output[3] != output[2]).any()
