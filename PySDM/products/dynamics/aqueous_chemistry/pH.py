from ...product import MomentProduct


class pH(MomentProduct):
    def __init__(self):
        super().__init__(
          name='pH',
          unit='',
          description='number-weighted pH',
          scale=None,
          range=None
        )

    def get(self):
        self.download_moment_to_buffer('pH', rank=1)
        return self.buffer
