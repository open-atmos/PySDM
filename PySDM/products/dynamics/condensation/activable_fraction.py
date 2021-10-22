from ...product import MomentProduct


class ActivableFraction(MomentProduct):
    def __init__(self):
        super().__init__(
            name="activable fraction",
            unit="1",
            description=""
        )

    def register(self, builder):
        super().register(builder)
        builder.request_attribute('critical supersaturation')

    def get(self, S_max):
        self.download_moment_to_buffer(
            'volume',
            rank=0,
            filter_range=(0, 1 + S_max / 100),
            filter_attr='critical supersaturation'
        )
        frac = self.buffer.copy()
        self.download_moment_to_buffer(
            'volume',
            rank=0
        )
        frac /= self.buffer
        return frac
