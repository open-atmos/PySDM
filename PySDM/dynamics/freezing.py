class Freezing:
    def __init__(self, *, singular=True):
        self.singular = singular
        self.enable = True

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

        if not self.enable:
            return

        self.core.bck.freeze(
            T_fz=self.core.particles['freezing temperature'],
            v_wet=self.core.particles['volume'],
            T=self.core.environment['T'],
            RH=self.core.environment['RH'],
            cell=self.core.particles['cell id']
        )

        self.core.particles.attributes['volume'].mark_updated()
