from PySDM.initialisation.sampling.spectral_sampling import Logarithmic
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si


def sampled_ccn_diameter_number_concentration_spectrum(
    n_sd: int = 11, size_range: tuple = (0.02 * si.um, 2 * si.um)
):
    return Logarithmic(
        spectrum=Lognormal(s_geom=1.4, m_mode=0.04 * si.um, norm_factor=100 / si.cm**3),
        size_range=size_range,
    ).sample(n_sd)
