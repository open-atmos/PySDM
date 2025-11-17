# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

from PySDM.backends import Numba, ThrustRTC


class TestCtorDefaults:
    @staticmethod
    def test_gpu_ctor_defaults():
        signature = inspect.signature(ThrustRTC.__init__)
        assert signature.parameters["verbose"].default is False
        assert signature.parameters["debug"].default is False
        assert signature.parameters["double_precision"].default is False
        assert signature.parameters["formulae"].default is None

    @staticmethod
    def test_cpu_ctor_defaults():
        signature = inspect.signature(Numba.__init__)
        assert signature.parameters["formulae"].default is None
