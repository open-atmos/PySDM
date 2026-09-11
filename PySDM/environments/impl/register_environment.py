"""decorator for environment classes
ensuring that their instances can be re-used with multiple builders"""

from copy import deepcopy


def _instantiate(self, particulator):
    copy = deepcopy(self)
    copy.register(particulator=particulator)
    return copy


def register_environment():
    def decorator(cls):
        if hasattr(cls, "instantiate"):
            if cls.instantiate is not _instantiate:
                raise AttributeError(
                    "decorated class has a different instantiate method"
                )
        else:
            setattr(cls, "instantiate", _instantiate)
        return cls

    return decorator
