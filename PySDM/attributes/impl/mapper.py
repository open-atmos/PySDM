"""
attribute name-variant-class mapping
"""

attributes = {}


def get_attribute_class(name, dynamics=None, formulae=None):
    if name not in attributes:
        raise ValueError(
            f"Unknown attribute name: {name}; valid names: {', '.join(sorted(attributes))}"
        )
    for cls, func in attributes[name].items():
        if func(dynamics, formulae):
            return cls
    assert False
