"""decorator for dynamics classes
ensuring that their instances can be re-used with multiple builders"""

from copy import deepcopy


def _instantiate(
    self,
    particulator,
):
    copy = deepcopy(self)
    copy.register(particulator=particulator)
    return copy


def register_dynamic():
    def decorator(cls):
        if hasattr(cls, "instantiate"):
            assert cls.instantiate is _instantiate
        else:
            setattr(cls, "instantiate", _instantiate)
        return cls

    return decorator
