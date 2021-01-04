"""
Created at 24.01.2020
"""


class Gravitational:

    def __init__(self):
        self.core = None
        self.tmp = None

    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('radius')
        builder.request_attribute('terminal velocity')
        self.tmp = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)


