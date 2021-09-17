"""
Wrapper class for triggering integration in the Eulerian advection solver used by the selected environment
"""


class EulerianAdvection:

    def __init__(self, solvers):
        self.solvers = solvers
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self):
        self.particulator.env.get_predicted('qv').download(self.particulator.env.get_qv(), reshape=True)
        self.particulator.env.get_predicted('thd').download(self.particulator.env.get_thd(), reshape=True)
        self.solvers()
