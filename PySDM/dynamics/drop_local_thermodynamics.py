from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class DropLocalThermodynamics:
    def __init__(self):
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.particulator.observers.append(self)
        assert "AmbientThermodynamics" in self.particulator.dynamics
        self.ambient_thermodynamics = self.particulator.dynamics[
            "AmbientThermodynamics"
        ]
        for var in ("water vapour mixing ratio", "dry air potential temperature"):
            builder.request_attribute(f"dropwise {var}")
            builder.request_attribute(f"dropwise {var} tendency")

    def __call__(self):
        # perform relaxation
        self.particulator.drop_local_thermodynamics()

    def notify(self):
        self.particulator.apply_dropwise_thermodynamic_tendency()
