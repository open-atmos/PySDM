from PySDM.dynamics import Collision, Displacement, Freezing


class SpinUp:
    def __init__(self, particulator, spin_up_steps):
        self.spin_up_steps = spin_up_steps
        particulator.observers.append(self)
        self.particulator = particulator
        self.set(Collision, "enable", False)
        self.set(Displacement, "enable_sedimentation", False)
        self.set(Freezing, "enable", False)

    def notify(self):
        if self.particulator.n_steps == self.spin_up_steps:
            self.set(Collision, "enable", True)
            self.set(Displacement, "enable_sedimentation", True)
            self.set(Freezing, "enable", True)

    def set(self, dynamic, attr, value):
        key = dynamic.__name__
        if key in self.particulator.dynamics:
            setattr(self.particulator.dynamics[key], attr, value)
