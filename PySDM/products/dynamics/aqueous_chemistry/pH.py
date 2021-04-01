from ...product import MomentProduct
import PySDM.physics.formulae as phys


class pH(MomentProduct):
    def __init__(self, radius_range):
        self.radius_range = radius_range
        super().__init__(
          name='pH',
          unit='',
          description='number-weighted pH',
          scale=None,
          range=None
        )

    def get(self):
        self.download_moment_to_buffer('pH', rank=1,
                                       filter_range=(phys.volume(self.radius_range[0]),
                                                     phys.volume(self.radius_range[1])))
        return self.buffer
