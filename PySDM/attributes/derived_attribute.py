"""
Created at 11.05.2020
"""

from .attribute import Attribute


class DerivedAttribute(Attribute):

    def __init__(self, particles_builder, name, dependencies):
        super().__init__(particles_builder, name)
        self.dependencies = dependencies

    def update(self):
        if self.data is None:
            self.allocate()
        for dependency in self.dependencies:
            dependency.update()
        dependencies_timestamp = sum(dependency.timestamp for dependency in self.dependencies)
        if self.timestamp < dependencies_timestamp:
            self.timestamp = dependencies_timestamp
            self.recalculate()

    def recalculate(self):
        raise NotImplementedError()
