from PySDM.physics import constants as const
from PySDM.products.product import MomentProduct


class AerosolSpecificConcentration(MomentProduct):

    def __init__(self, radius_threshold):
        self.radius_threshold = radius_threshold
        super().__init__(
            name='n_a_mg',
            unit='mg-1',
            description='Aerosol specific concentration'
        )

    def get(self):
        self.download_moment_to_buffer('volume', rank=0,
                                       filter_range=[0, self.formulae.trivia.volume(self.radius_threshold)])
        result = self.buffer.copy()  # TODO #217
        self.download_to_buffer(self.core.environment['rhod'])
        result[:] /= self.core.mesh.dv
        result[:] /= self.buffer
        const.convert_to(result, const.si.milligram**-1)
        return result
