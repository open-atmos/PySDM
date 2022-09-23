# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

import pytest

from PySDM.dynamics.collisions import Breakup, Coalescence, Collision


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class TestDefaults:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize("dynamic_class", (Collision, Breakup, Coalescence))
    def test_collision_adaptive_default(dynamic_class):
        assert get_default_args(dynamic_class.__init__)["adaptive"] is True
