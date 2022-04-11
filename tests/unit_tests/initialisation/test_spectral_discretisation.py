# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling, spectro_glacial_sampling
from PySDM.initialisation.spectra.lognormal import Lognormal

m_mode = 0.5e-5
n_part = 256 * 16
s_geom = 1.5
spectrum = Lognormal(n_part, m_mode, s_geom)
m_range = (0.1e-6, 100e-6)
formulae = Formulae(
    freezing_temperature_spectrum="Niemand_et_al_2012",
    constants={"NIEMAND_A": -0.517, "NIEMAND_B": 8.934},
)


@pytest.mark.parametrize(
    "discretisation",
    [
        pytest.param(spectral_sampling.Linear(spectrum, m_range)),
        pytest.param(spectral_sampling.Logarithmic(spectrum, m_range)),
        pytest.param(spectral_sampling.ConstantMultiplicity(spectrum, m_range)),
        pytest.param(spectral_sampling.UniformRandom(spectrum, m_range)),
        # TODO #599
    ],
)
def test_spectral_discretisation(discretisation):
    # Arrange
    n_sd = 100000

    # Act
    if isinstance(discretisation, spectro_glacial_sampling.SpectroGlacialSampling):
        m, _, __, n = discretisation.sample(n_sd)
    else:
        m, n = discretisation.sample(n_sd)

    # Assert
    assert m.shape == n.shape
    assert n.shape == (n_sd,)
    assert np.min(m) >= m_range[0]
    assert np.max(m) <= m_range[1]
    actual = np.sum(n)
    desired = spectrum.cumulative(m_range[1]) - spectrum.cumulative(m_range[0])
    quotient = actual / desired
    np.testing.assert_approx_equal(actual=quotient, desired=1.0, significant=2)
