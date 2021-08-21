class Freezing:
    def __init__(self, *, singular=True):
        self.singular = singular

    def register(self, builder):
        if self.singular:
            builder.request_attribute("freezing temperature")
        else:
            raise NotImplementedError()
        builder.request_attribute("spheroid mass")
        self.core = builder.core

    def __call__(self):
        if 'Coalescence' in self.core.dynamics:
            raise NotImplementedError("handling T_fz during collisions not implemented yet")  # TODO #594
