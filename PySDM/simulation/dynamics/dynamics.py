"""
Created at 05.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Dynamics:
    def __init__(self, particles):
        self.particles = particles
        self.__instances = {}

    def register(self,  dynamic_class, params: dict):
        instance = (dynamic_class(self.particles, **params))
        self.__instances[str(dynamic_class)] = instance
        self.particles.register_products(instance)

    def step_all(self):
        for dynamic in self.__instances.values():
            dynamic()
