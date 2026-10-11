# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from contextlib import nullcontext
from unittest import mock
import warnings
import inspect
import pytest
import numba

from PySDM.backends import Numba, ThrustRTC


class TestCtorDefaultsAndWarnings:
    @staticmethod
    def test_gpu_ctor_defaults():
        signature = inspect.signature(ThrustRTC.__init__)
        assert signature.parameters["verbose"].default is False
        assert signature.parameters["debug"].default is False
        assert signature.parameters["double_precision"].default is False
        assert signature.parameters["formulae"].default is None
        assert signature.parameters["n_dims"].default == 0

    @staticmethod
    def test_cpu_ctor_defaults():
        signature = inspect.signature(Numba.__init__)
        assert signature.parameters["formulae"].default is None
        assert signature.parameters["n_dims"].default == 0

    @staticmethod
    def test_jax_ctor_defaults():
        signature = inspect.signature(Numba.__init__)
        assert signature.parameters["formulae"].default is None
        assert signature.parameters["n_dims"].default == 0

    @staticmethod
    @mock.patch("PySDM.backends.numba.prange", new=range)
    def test_check_numba_threading_warning():
        if numba.config.DISABLE_JIT:  # pylint: disable=no-member
            pytest.skip()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_info:
                Numba(n_dims=1)
            assert exc_info.match(r"^Numba threading enabled but does not work")

    @staticmethod
    @pytest.mark.parametrize(
        "setting, outcome, context",
        (
            (True, True, nullcontext()),
            (False, False, nullcontext()),
            (
                True,
                False,
                mock.patch(
                    "numba.parfors.parfor.ensure_parallel_support",
                    side_effect=numba.core.errors.UnsupportedParforsError("Mock"),
                ),
            ),
            (
                False,
                False,
                mock.patch(
                    "numba.parfors.parfor.ensure_parallel_support",
                    side_effect=numba.core.errors.UnsupportedParforsError("Mock"),
                ),
            ),
        ),
    )
    def test_numba_override_jit_flags_parallel(setting, outcome, context):
        with context:
            backend = Numba(override_jit_flags={"parallel": setting})
        assert backend.default_jit_flags["parallel"] == outcome
