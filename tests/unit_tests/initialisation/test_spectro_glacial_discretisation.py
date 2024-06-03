# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.initialisation.sampling import spectro_glacial_sampling
from PySDM.initialisation.spectra.lognormal import Lognormal
from PySDM.physics import si

formulae = Formulae(
    freezing_temperature_spectrum="Niemand_et_al_2012",
    constants={"NIEMAND_A": -0.517, "NIEMAND_B": 8.934},
)
spectrum = Lognormal(
    norm_factor=1,
    m_mode=formulae.trivia.sphere_surface(diameter=0.74 * si.um),
    s_geom=np.exp(0.25),
)
m_range = (
    formulae.trivia.sphere_surface(diameter=0.01 * si.um),
    formulae.trivia.sphere_surface(diameter=100.0 * si.um),
)


@pytest.mark.parametrize(
    "discretisation",
    (
        pytest.param(
            spectro_glacial_sampling.SpectroGlacialSampling(
                freezing_temperature_spectrum=formulae.freezing_temperature_spectrum,
                insoluble_surface_spectrum=spectrum,
            )
        ),
    ),
)
def test_spectral_discretisation(discretisation, backend_instance):
    # Arrange
    n_sd = 100000
    backend = backend_instance

    # Act
    freezing_temperatures, immersed_surfaces, n = discretisation.sample(
        n_sd=n_sd, backend=backend
    )

    # Assert
    assert n.shape == (n_sd,)
    actual = np.sum(n)
    desired = spectrum.cumulative(m_range[1]) - spectrum.cumulative(m_range[0])
    quotient = actual / desired
    np.testing.assert_approx_equal(actual=quotient, desired=1.0, significant=2)

    assert (formulae.constants.T0 - 50 < freezing_temperatures).all()
    assert (freezing_temperatures < formulae.constants.T0).all()

    assert (immersed_surfaces < m_range[1]).all()
    assert (immersed_surfaces > m_range[0]).all()
