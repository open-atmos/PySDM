import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics.constants import PER_CENT, si, in_unit

def test_twomey_and_wojciechowski_1969_fig1(plot=True):
    """[Twomey, Wojciechowski 1969](https://doi.org/10.1175/1520-0469(1969)26%3C648:OOTGVO%3E2.0.CO;2)
    """
    #arange
    for k, N in zip([0.5,0.7], [100,500]):
        formulae = Formulae(ccn_activation_spectrum="Twomey1959",constants={"TWOMEY_K": k, "TWOMEY_N0": N/si.cm**3})
        supersaturation = np.logspace(np.log10(.2), np.log10(9))*PER_CENT
        #act
        activated_nuclei_concentration = formulae.ccn_activation_spectrum.ccn_concentration(saturation_ratio=supersaturation+1)
        #plot
        pyplot.plot(in_unit(supersaturation,PER_CENT),
                    in_unit(activated_nuclei_concentration,si.cm**-3),
                    label=f"{k=}"
                )
    pyplot.xlim(0.1,10)
    pyplot.ylim(1,1000)
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlabel("Percent supersaturation")
    pyplot.ylabel("Nuclei [cm$^{-3}$]")
    pyplot.grid()
    pyplot.legend()
    pyplot.show()
    #assert
