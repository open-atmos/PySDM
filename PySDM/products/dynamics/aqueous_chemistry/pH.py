from ...product import MomentProduct


class pH(MomentProduct):
    def __init__(self):
        super().__init__(
          name='pH',
          unit='TODO',
          description='pH',
          scale=None,
          range=None
        )

    def get(self):
        self.download_moment_to_buffer('pH', rank=1)
#        self.buffer[:] /= self.core.mesh.dv
        return self.buffer