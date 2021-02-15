from chempy import Substance


class AqueousChemistry:
    def __init__(self, environment_amount):
        self.environment_amount = environment_amount

    def register(self, builder):
        mesh = builder.core.mesh
        if mesh.dimension != 0:
            raise NotImplementedError()

