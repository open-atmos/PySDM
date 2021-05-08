from PySDM_examples.Partridge_et_al_2011.settings import (
    MarineArctic, MarineAverage, RuralContinental, PollutedContinental
)
from PySDM_examples.Partridge_et_al_2011.simulation import Simulation
import re


def test_fig_3(plot=False):
    # Arrange
    runs = []
    for settings_class in (MarineArctic, MarineAverage, RuralContinental, PollutedContinental):
        runs.append(Simulation(settings_class()))

    # Act
    # TODO #491

    # Plot
    if plot:
        from matplotlib import pyplot
        for run in runs:
            pyplot.plot((0, 1), color=run.settings.color, label=_name(run.settings))
        pyplot.ylabel("dN/dlogD$_p$ cm$^{-3}$")
        pyplot.xlabel("Diameter nm")
        pyplot.legend()
        pyplot.show()

    # Assert

_name_pattern = re.compile(r'(?<!^)(?=[A-Z])')

def _name(settings_instance):
    return _name_pattern.sub(' ', str(settings_instance.__class__.__name__)).lower()
