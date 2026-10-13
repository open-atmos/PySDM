"""decorator for environment classes
ensuring that their instances cannot be re-used in multiple particulators"""

from copy import deepcopy


def _instantiate(self, particulator):
    """Creating a deep copy to facilitate environment reuse in user code,
    but exluding backend (and the formulae within) to avoid re-JIT-compilations"""
    backend = self.backend
    self.backend = None
    copy = deepcopy(self)
    copy.backend = backend
    self.backend = backend
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
