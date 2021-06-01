"""
average pH (averaging after or before taking the logarithm in pH definition)
with weighting either by number or volume
"""
from ...product import MomentProduct


class pH(MomentProduct):
    def __init__(self, radius_range, weighting='volume', attr='conc_H'):
        assert attr in ('pH', 'moles_H', 'conc_H')
        self.attr = attr

        if weighting == 'number':
            self.weighting_rank = 0
        elif weighting == 'volume':
            self.weighting_rank = 1
        else:
            raise NotImplementedError()

        self.radius_range = radius_range
        super().__init__(
            name='pH_' + attr + '_' + weighting + '_weighted',
            unit='',
            description='number-weighted pH'
        )

    def register(self, builder):
        builder.request_attribute('conc_H')
        super().register(builder)

    def get(self):
        self.download_moment_to_buffer(self.attr, rank=1,
                                       filter_range=(self.formulae.trivia.volume(self.radius_range[0]),
                                                     self.formulae.trivia.volume(self.radius_range[1])),
                                       weighting_attribute='volume', weighting_rank=self.weighting_rank)
        if self.attr == 'conc_H':
            self.buffer[:] = self.formulae.trivia.H2pH(self.buffer[:])
        elif self.attr == 'pH':
            pass
        else:
            raise NotImplementedError()

        return self.buffer
