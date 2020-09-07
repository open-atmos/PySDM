"""
Created at 29.11.2019
"""


class EulerianAdvection:

    def __init__(self):
        self.core = None

    def register(self, builder):
        self.core = builder.core
        self.core.observers.append(self)
        self.notify()

    def __call__(self):
        self.core.env.get_predicted('qv').download(self.core.env.get_qv(), reshape=True)
        self.core.env.get_predicted('thd').download(self.core.env.get_thd(), reshape=True)
        self.core.env.step()

    def notify(self):
        self.core.env.sync()
