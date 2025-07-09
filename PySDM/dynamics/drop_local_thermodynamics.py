from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class DropLocalThermodynamics:
    def __init__(self):
        self.particulator = None

    def register(self, builder):
        builder.request_attribute("drop-local water vapour mixing ratio")
        self.particulator = builder.particulator

    def __call__(self):
        self.particulator.drop_local_thermodynamics()
