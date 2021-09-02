class Freezing:
    def __init__(self, *, singular=True, r=None):
        self.singular = singular
        self.enable = True
        self.r = r
        self.rand = None
        self.rng = None

    def register(self, builder):
        self.core = builder.core

        builder.request_attribute("volume")
        if self.singular:
            builder.request_attribute("freezing temperature")
        else:
            builder.request_attribute("nucleation sites")
            self.rand = self.core.Storage.empty(self.core.n_sd, dtype=float)
            self.rng = self.core.Random(self.core.n_sd, self.core.bck.formulae.seed)

    def __call__(self):
        if 'Coalescence' in self.core.dynamics:
            raise NotImplementedError("handling T_fz during collisions not implemented yet")  # TODO #594

        if not self.enable:
            return

        if self.singular:
            self.core.bck.freeze_singular(
                T_fz=self.core.particles['freezing temperature'],
                v_wet=self.core.particles['volume'],
                T=self.core.environment['T'],
                RH=self.core.environment['RH'],
                cell=self.core.particles['cell id']
            )
        else:
            self.rand.urand(self.rng)
            self.core.bck.freeze_time_dependent(
                rand=self.rand,
                nucleation_sites=self.core.particles['nucleation sites'],
                volume=self.core.particles['volume'],
                r=self.r,
                dt=self.core.dt
            )

        self.core.particles.attributes['volume'].mark_updated()
