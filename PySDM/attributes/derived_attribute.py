"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .attribute import Attribute


class DerivedAttribute(Attribute):

    def __init__(self, particles_builder, name, dependencies):
        super().__init__(particles_builder, name)
        self.dependencies = dependencies

    def get(self):
        self.update()
        return self.data

    def update(self):
        if self.data is None:
            self.allocate()
        dependencies_timestamp = sum(dependency.timestamp for dependency in self.dependencies)
        if self.timestamp < dependencies_timestamp:
            self.timestamp = dependencies_timestamp
            self.recalculate()

    def recalculate(self):
        raise NotImplementedError()
