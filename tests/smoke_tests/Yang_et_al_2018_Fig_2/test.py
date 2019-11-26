from examples.Yang_et_al_2018.example import Simulation, Setup
from PySDM.simulation.physics.constants import si
from PySDM.utils import Physics


def test(plot=True):
    simulation = Simulation()
    dry_volume = simulation.particles.state.get_backend_storage('x') #TODO: to_ndarray
    rd = Physics.x2r(dry_volume) / si.nanometre
    nd = simulation.particles.state.n # TODO: to_ndarray

    rd = rd[::-1]
    print(rd)
    assert round(rd[  1-1], 0) == 503
    assert round(rd[ 10-1], 0) == 355
    assert round(rd[ 50-1], 1) == 75.3
    assert round(rd[100-1], 1) == 10.8

    dr = rd[1:] - rd[0:-1]
    dn_dr = (nd[0:-1] * Setup.rho / dr) #.to(1/si.centimetre**3 / si.nanometre)

    # from fig. 1b
    # TODO
    # assert 1e-2 < dn_dr[0] < 1e-1
    # assert 1e1 < max(dn_dr) < 1e2
    # assert dn_dr[-1] < 1e-9
