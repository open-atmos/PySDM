"""
logic around `PySDM.attributes.impl.derived_attribute.DerivedAttribute` - the parent class
for all derived attributes
"""

from .attribute import Attribute


class DerivedAttribute(Attribute):
    def __init__(self, builder, name, dependencies):
        assert len(dependencies) > 0
        super().__init__(builder, name)
        self.dependencies = dependencies

    def update(self):
        for dependency in self.dependencies:
            dependency.update()
        dependencies_timestamp = sum(
            dependency.timestamp for dependency in self.dependencies
        )
        if self.timestamp < dependencies_timestamp:
            self.timestamp = dependencies_timestamp
            self.recalculate()

    def recalculate(self):
        raise NotImplementedError()

    def mark_updated(self):
        raise AssertionError()
