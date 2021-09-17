from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class CriticalVolume(DerivedAttribute):
    def __init__(self, builder):
        self.cell_id = builder.get_attribute('cell id')
        self.v_dry = builder.get_attribute('dry volume')
        self.v_wet = builder.get_attribute('volume')
        self.kappa = builder.get_attribute('kappa')
        self.f_org = builder.get_attribute('dry volume organic fraction')
        self.environment = builder.particulator.environment
        self.particles = builder.particulator
        dependencies = [self.v_dry, self.v_wet, self.cell_id]
        super().__init__(builder, name='critical volume', dependencies=dependencies)

    def recalculate(self):
        self.particulator.bck.critical_volume(
            self.data,
            kappa=self.kappa.get(),
            f_org=self.f_org.get(),
            v_dry=self.v_dry.get(),
            v_wet=self.v_wet.get(),
            T=self.environment['T'],
            cell=self.cell_id.get()
        )
