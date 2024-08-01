from copy import deepcopy


def _instantiate(self, *, builder, buffer):
    copy = deepcopy(self)
    copy.set_buffer(buffer)
    copy.register(builder)
    return copy


def register_product():
    def decorator(cls):
        if hasattr(cls, "instantiate"):
            assert cls.instantiate is _instantiate
        else:
            setattr(cls, "instantiate", _instantiate)
        return cls

    return decorator
