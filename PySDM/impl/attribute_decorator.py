from PySDM.impl.camel_case import camel_case_to_words


def attribute(*, name=None, variant=None, dummy_default=False, warn=False):
    if variant is None:
        assert name is None
    if dummy_default:
        assert variant is not None

    def decorator(cls):
        from PySDM.attributes.impl.mapper import attributes
        from PySDM.attributes.impl.dummy_attribute import make_dummy_attribute_factory

        key = name or camel_case_to_words(cls.__name__)

        if key not in attributes:
            attributes[key] = {}
        elif cls in attributes[key]:
            raise ValueError(f"attribute {key} already exists!")
        attributes[key][cls] = variant or (lambda _, __: cls)
        if dummy_default:
            attributes[key][make_dummy_attribute_factory(key, warn=warn)] = (
                lambda _, __: not variant(_, __)
            )
        return cls

    return decorator
