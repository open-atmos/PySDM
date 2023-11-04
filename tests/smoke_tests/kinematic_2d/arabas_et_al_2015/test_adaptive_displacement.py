# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998 import Simulation

from PySDM import Formulae
from PySDM.backends import CPU
from PySDM.physics import si
from PySDM.products import SuperDropletCountPerGridbox

from .dummy_storage import DummyStorage


@pytest.mark.parametrize(
    "rtol",
    (
        pytest.param(None, marks=pytest.mark.xfail(strict=True)),
        pytest.param(1e-0, marks=pytest.mark.xfail(strict=True)),
        pytest.param(1e-1, marks=pytest.mark.xfail(strict=True)),
        pytest.param(1e-2),
    ),
)
def test_adaptive_displacement(rtol, plot=False):
    # Arrange
    settings = Settings(formulae=Formulae(seed=666))
    settings.dt = 5 * si.s
    settings.grid = (10, 10)
    settings.n_sd_per_gridbox = 10
    settings.rhod_w_max = 10 * si.m / si.s * si.kg / si.m**3

    settings.simulation_time = 1000 * si.s
    settings.spin_up_time = settings.simulation_time
    settings.output_interval = settings.simulation_time
    if rtol is not None:
        settings.displacement_adaptive = True
        settings.displacement_rtol = rtol
    else:
        settings.displacement_adaptive = False
    settings.processes["condensation"] = False

    storage = DummyStorage()
    simulation = Simulation(settings, storage, SpinUp=SpinUp, backend_class=CPU)
    simulation.reinit(products=[SuperDropletCountPerGridbox()])

    # Act
    simulation.run()
    sd_count = simulation.products["super droplet count per gridbox"].get()

    # Plot
    pyplot.imshow(
        sd_count.T, origin="lower", extent=(0, settings.grid[0], 0, settings.grid[1])
    )
    cbar = pyplot.colorbar()
    cbar.set_label("#SD / cell")
    pyplot.clim(0, 2 * settings.n_sd_per_gridbox)
    pyplot.title(
        f"adaptive: {settings.displacement_adaptive} {f'(rtol={rtol})' if rtol else ''}"
    )
    pyplot.xlabel("x/dx")
    pyplot.ylabel("z/dz")
    pyplot.xticks(np.arange(settings.grid[0] + 1))
    pyplot.yticks(np.arange(settings.grid[1] + 1))
    if plot:
        pyplot.show()

    # Assert
    if rtol is not None:
        assert 1 < simulation.particulator.dynamics["Displacement"]._n_substeps < 50
    assert np.count_nonzero(sd_count) == np.product(settings.grid)
    assert np.std(sd_count) < settings.n_sd_per_gridbox / 2.5
    assert np.max(sd_count) < 2.5 * settings.n_sd_per_gridbox
