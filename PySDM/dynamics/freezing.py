class Freezing:
    def __init__(self, *, singular=True, J_het=None):
        self.singular = singular
        self.enable = True
        self.J_het = J_het
        self.rand = None
        self.rng = None

    def register(self, builder):
        self.particulator = builder.particulator

        builder.request_attribute("volume")
        if self.singular:
            assert self.J_het is None
            builder.request_attribute("freezing temperature")
        else:
            assert self.J_het is not None
            builder.request_attribute("immersed surface area")
            self.rand = self.particulator.Storage.empty(self.particulator.n_sd, dtype=float)
            self.rng = self.particulator.Random(self.particulator.n_sd, self.particulator.bck.formulae.seed)

    def __call__(self):
        if 'Coalescence' in self.particulator.dynamics:
            raise NotImplementedError("handling T_fz during collisions not implemented yet")  # TODO #594

        if not self.enable:
            return

        if self.singular:
            self.particulator.bck.freeze_singular(
                T_fz=self.particulator.attributes['freezing temperature'],
                v_wet=self.particulator.attributes['volume'],
                T=self.particulator.environment['T'],
                RH=self.particulator.environment['RH'],
                cell=self.particulator.attributes['cell id']
            )
        else:
            self.rand.urand(self.rng)
            self.particulator.bck.freeze_time_dependent(
                rand=self.rand,
                immersed_surface_area=self.particulator.attributes['immersed surface area'],
                volume=self.particulator.attributes['volume'],
                J_het=self.J_het,
                dt=self.particulator.dt
            )

        self.particulator.attributes.attributes['volume'].mark_updated()
