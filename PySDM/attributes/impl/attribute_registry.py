""" definition of decorator used to register PySDM attribute classes """

import warnings

from PySDM.impl.camel_case import camel_case_to_words

_ATTRIBUTES_REGISTRY = {}


def _make_dummy_attribute_factory(name):
    # pylint: disable=import-outside-toplevel
    from PySDM.attributes.impl.dummy_attribute import DummyAttribute

    def _factory(builder):
        return DummyAttribute(builder, name=name)

    return _factory


def register_attribute(*, name=None, variant=None, dummy_default=False, warn=False):
    if variant is not None:
        assert name is not None
    if dummy_default:
        assert variant is not None
    if warn:
        assert dummy_default

    def decorator(cls):
        key = name or camel_case_to_words(cls.__name__)

        if key not in _ATTRIBUTES_REGISTRY:
            _ATTRIBUTES_REGISTRY[key] = {}
        elif cls in _ATTRIBUTES_REGISTRY[key]:
            raise ValueError(f"attribute {key} already exists!")
        _ATTRIBUTES_REGISTRY[key][cls] = variant or (lambda _, __: cls)
        if dummy_default:
            _ATTRIBUTES_REGISTRY[key][_make_dummy_attribute_factory(key)] = (
                lambda _, __: (
                    warnings.warn(
                        f"dummy implementation used for requested attribute named '{name}'"
                    )
                    if warn
                    else None
                )
                is None
                and not variant(_, __)
            )
        return cls

    return decorator


def get_attribute_class(name, dynamics=None, formulae=None):
    if name not in _ATTRIBUTES_REGISTRY:
        raise ValueError(
            f"Unknown attribute name: {name};"
            " valid names: {', '.join(sorted(_ATTRIBUTES_REGISTRY))}"
        )
    for cls, func in _ATTRIBUTES_REGISTRY[name].items():
        if func(dynamics, formulae):
            return cls
    assert False
