import inspect
from PySDM.dynamics.collisions import Collision

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

class Test_defaults:
    @staticmethod
    def test_collision_adaptive_default():
        assert get_default_args(Collision.__init__)['adaptive'] == True
