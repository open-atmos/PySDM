from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.settings import Settings, w_avgs
from PySDM.physics.constants import si, rho_STP, convert_to
import pytest
import numpy as np


@pytest.mark.parametrize("settings_idx", range(len(w_avgs)))
def test_event_rates(settings_idx):
    # Arrange
    settings = Settings(
        w_avg=setups[settings_idx].w_avg,
        N_STP=setups[settings_idx].N_STP,
        r_dry=setups[settings_idx].r_dry,
        mass_of_dry_air=1 * si.kg
    )
    settings.n_output = 50
    simulation = Simulation(settings)

    # Act
    output = simulation.run()

    # Assert
    rip = np.asarray(output['ripening_rate'])
    act = np.asarray(output['activating_rate'])
    dea = np.asarray(output['deactivating_rate'])
    act_max = np.full((1,), settings.n_in_dv / simulation.particulator.dt / rho_STP)
    convert_to(act_max, 1/si.mg)
    assert (rip == 0).all()
    assert (act > 0).any()
    assert (dea > 0).any()
    assert 0 < max(act) < act_max[0]
    assert 0 < max(dea) < act_max[0]
