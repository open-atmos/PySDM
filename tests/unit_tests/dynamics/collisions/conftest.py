# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.dynamics.collisions.collision import DEFAULTS, Coalescence
from PySDM.environments import Box

from ....unit_tests.dummy_particulator import DummyParticulator


def backend_fill(array, value, odd_zeros=False):
    if odd_zeros:
        if isinstance(value, np.ndarray):
            full_ndarray = insert_zeros(value[::2]).astype(np.float64)
        else:
            full_ndarray = np.full(array.shape[0] // 2, value).astype(np.float64)
            full_ndarray = insert_zeros(full_ndarray)
            if array.shape[0] % 2 != 0:
                full_ndarray = np.concatenate((full_ndarray, np.zeros(1)))
    else:
        full_ndarray = np.full(array.shape, value).astype(np.float64)

    array.upload(full_ndarray)


def insert_zeros(array):
    result = (
        np.concatenate((array, np.zeros_like(array))).reshape(2, -1).flatten(order="F")
    )
    return result


def get_dummy_particulator_and_coalescence(
    backend,
    n_length,
    optimized_random=False,
    environment=None,
    substeps=1,
    attributes: dict | None = None,
    requested_attributes: tuple = (),
):
    backend = backend(
        double_precision=True,
        formulae=Formulae(
            collision_kernel="ConstantK", constants={"KERNEL_CONSTANTK_a": 0}
        ),
    )
    env = environment or Box(
        backend=backend,
        dv=1,
        dt=DEFAULTS.dt_coal_range[1],
    )

    coalescence = Coalescence(
        optimized_random=optimized_random,
        substeps=substeps,
        adaptive=False,
    )

    particulator = DummyParticulator(
        n_sd=n_length,
        environment=env,
        attributes=attributes,
        requested_attributes=requested_attributes,
        dynamics=(coalescence,),
    )
    coalescence = particulator.dynamics["Collision"]
    return particulator, coalescence


__x__ = {
    "ones_2": pytest.param(np.array([1.0, 1.0])),
    "random_2": pytest.param(np.array([4.0, 2.0])),
}


@pytest.fixture(params=[__x__["ones_2"], __x__["random_2"]], name="v_2")
def v_2_fixture(request):
    return request.param


@pytest.fixture(params=[__x__["ones_2"], __x__["random_2"]], name="T_2")
def T_2_fixture(request):
    return request.param


__n__ = {
    "1_1": pytest.param(np.array([1, 1])),
    "5_1": pytest.param(np.array([5, 1])),
    "5_3": pytest.param(np.array([5, 3])),
}


@pytest.fixture(params=[__n__["1_1"], __n__["5_1"], __n__["5_3"]], name="n_2")
def n_2_fixture(request):
    return request.param
