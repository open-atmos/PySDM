"""
Created at 07.09.2020
"""


class LagrangianAdvection:

    def __init__(self):
        self.core = None

    def register(self, builder):
        self.core = builder.core

    def __call__(self):
        self.core.env.sync()
