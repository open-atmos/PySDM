import numpy as np
from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct


class ParticlesConcentration(MomentProduct):

    def __init__(self, radius_range=(0, np.inf), specific=False):
        self.radius_range = radius_range
        self.specific = specific
        super().__init__(
            name=f'n_part_{"mg" if specific else "cm3"}',
            unit='mg-1' if specific else 'cm-3',
            description=f'Particle {"specific " if specific else ""}concentration'
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0,
                                       filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                                                     self.formulae.trivia.volume(self.radius_range[1])))
        self.buffer[:] /= self.particulator.mesh.dv
        if self.specific:
            result = self.buffer.copy()  # TODO #217
            self.download_to_buffer(self.particulator.environment['rhod'])
            result[:] /= self.buffer
        else:
            result = self.buffer
        const.convert_to(result, const.si.mg**-1 if self.specific else const.si.centimetre**-3)
        return result


class AerosolConcentration(ParticlesConcentration):

    def __init__(self, radius_threshold):
        super().__init__((0, radius_threshold))
        self.name = 'n_a_cm3'
        self.description = 'Aerosol particles concentration'


class CloudDropletConcentration(ParticlesConcentration):

    def __init__(self, radius_range):
        super().__init__(radius_range)
        self.name = 'n_c_cm3'
        self.description = 'Cloud droplets concentration'


class DrizzleConcentration(ParticlesConcentration):

    def __init__(self, radius_threshold):
        super().__init__((radius_threshold, np.inf))
        self.name = 'n_d_cm3'
        self.description = 'Drizzle droplets concentration'
