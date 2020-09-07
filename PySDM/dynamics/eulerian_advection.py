"""
Created at 29.11.2019
"""


class EulerianAdvection:

    def __init__(self):
        self.core = None

    def register(self, builder):
        self.core = builder.core

    def __call__(self):
        self.core.env.get_predicted('qv').download(self.core.env.get_qv(), reshape=True)
        self.core.env.get_predicted('thd').download(self.core.env.get_thd(), reshape=True)
        self.core.env.step()
