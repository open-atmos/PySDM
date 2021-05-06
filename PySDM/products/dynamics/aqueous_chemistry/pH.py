from ...product import MomentProduct


class pH(MomentProduct):
    def __init__(self, radius_range):
        self.radius_range = radius_range
        super().__init__(
          name='pH',
          unit='',
          description='number-weighted pH'
        )

    def get(self):
        self.download_moment_to_buffer('pH', rank=1,
                                       filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                                                     self.formulae.trivia.volume(self.radius_range[1])))
        return self.buffer
