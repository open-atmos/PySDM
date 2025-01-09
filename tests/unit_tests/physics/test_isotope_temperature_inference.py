""" checking values in Fig. 3 in [Picciotto et al. 1960](https://doi.org/10.1038/187857a0) """

import numpy as np
from matplotlib import pyplot
from PySDM import Formulae
from PySDM.physics.constants import PER_MILLE
from PySDM.physics import in_unit


def test_fig_3(plot=False):
    # arrange
    formulae = Formulae(isotope_temperature_inference="PicciottoEtAl1960")
    delta_18O = np.linspace(-35, -9) * PER_MILLE
    T0 = formulae.constants.T0

    # act
    temperature = formulae.isotope_temperature_inference.temperature_from_delta_18O(
        delta_18O
    )

    # plot
    pyplot.figure(figsize=(5, 6))
    for side in ("top", "right"):
        pyplot.gca().spines[side].set_visible(False)
    pyplot.plot(temperature - T0, in_unit(delta_18O, PER_MILLE), color="k")

    pyplot.xlabel("temperature [°C]")
    pyplot.xlim(-45, 2)
    pyplot.xticks(range(-45, 2, 5))

    pyplot.ylabel("δ$^{18}$O [‰]")
    pyplot.ylim(in_unit(delta_18O[0], PER_MILLE), in_unit(delta_18O[-1], PER_MILLE))
    pyplot.yticks(range(-35, -9, 5))

    pyplot.grid(color="k")
    pyplot.savefig("pysdm_fig3.pdf")
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    monotonic_temperature = (np.diff(temperature) > 0).all()
    assert monotonic_temperature
    assert -32 < (temperature[0] - T0) < -31
    assert -3 < (temperature[-1] - T0) < -2
