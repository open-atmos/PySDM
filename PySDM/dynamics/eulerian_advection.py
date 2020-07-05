"""
Created at 29.11.2019
"""

from PySDM.environments import MoistEulerianInterface
from PySDM.builder import Builder


class EulerianAdvection:

    def __init__(self):
        self.core = None

    def register(self, builder):
        self.core = builder.core

    def __call__(self):
        env: MoistEulerianInterface = self.core.environment
        env.get_predicted('qv').download(env.get_qv().ravel())
        env.get_predicted('thd').download(env.get_thd().ravel())

        env.step()
