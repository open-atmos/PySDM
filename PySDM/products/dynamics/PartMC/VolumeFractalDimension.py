from ...product import Product

class VolumeFractalDimension(Product):
    def __init__(self):
        super().__init__(
            name='PMC_fractal_surface_frac_dim',
            unit='1',
            description=f'PartMC: volume fractal dimension'
        )
