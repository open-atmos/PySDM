class Freezing:
    def __init__(self, *, singular=True, J_het=None):
        self.singular = singular
        self.enable = True
        self.J_het = J_het
        self.rand = None
        self.rng = None

    def register(self, builder):
        self.core = builder.core

        builder.request_attribute("volume")
        if self.singular:
            assert self.J_het is None
            builder.request_attribute("freezing temperature")
        else:
            assert self.J_het is not None
            builder.request_attribute("immersed surface area")
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
                immersed_surface_area=self.core.particles['immersed surface area'],
                volume=self.core.particles['volume'],
                J_het=self.J_het,
                dt=self.core.dt
            )

        self.core.particles.attributes['volume'].mark_updated()
