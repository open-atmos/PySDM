import warnings
from PySDM.impl.camel_case import camel_case_to_words


def register_attribute(*, name=None, variant=None, dummy_default=False, warn=False):
    if variant is not None:
        assert name is not None
    if dummy_default:
        assert variant is not None
    if warn:
        assert dummy_default

    def decorator(cls):
        # pylint: disable=import-outside-toplevel
        from PySDM.attributes.impl.mapper import attributes
        from PySDM.attributes.impl.dummy_attribute import make_dummy_attribute_factory

        key = name or camel_case_to_words(cls.__name__)

        if key not in attributes:
            attributes[key] = {}
        elif cls in attributes[key]:
            raise ValueError(f"attribute {key} already exists!")
        attributes[key][cls] = variant or (lambda _, __: cls)
        if dummy_default:
            attributes[key][make_dummy_attribute_factory(key)] = lambda _, __: (
                warnings.warn(
                    f"dummy implementation used for requested attribute named '{name}'"
                )
                if warn
                else None
            ) is None and not variant(_, __)
        return cls

    return decorator
