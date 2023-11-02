# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

import pytest

from PySDM.backends import CPU, GPU


class TestCtorDefaults:
    @staticmethod
    def test_gpu_ctor_defaults():
        signature = inspect.signature(GPU.__init__)
        assert signature.parameters["verbose"].default is False
        assert signature.parameters["debug"].default is False
        assert signature.parameters["double_precision"].default is False
        assert signature.parameters["formulae"].default is None

    @staticmethod
    def test_cpu_ctor_defaults():
        signature = inspect.signature(CPU.__init__)
        assert signature.parameters["formulae"].default is None

    @staticmethod
    def test_formulae_unchangeable_after_disabling_setattr(backend_class):
        sut = backend_class()
        sut.disable_setattr()
        with pytest.raises(AssertionError):
            sut.formulae = 123
