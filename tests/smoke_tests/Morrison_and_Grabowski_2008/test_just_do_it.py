from PySDM_examples.Szumowski_et_al_1998 import Simulation, Storage
from PySDM_examples.Morrison_and_Grabowski_2008 import ColdCumulus
from contextlib import AbstractContextManager
import numpy as np


class FlowFieldAsserts(AbstractContextManager):
    def __init__(self, simulation):
        self.simulation = simulation
        self.panic = None

    def set_percent(self, percent):
        if percent == 0:
            return
        advector = None
        for solver in self.simulation.particulator.dynamics['EulerianAdvection'].solvers.mpdatas.values():
            assert advector is None or advector is solver.advector
            advector = solver.advector
        np.testing.assert_allclose(advector.get_component(1)[:, 0], 0)
        np.testing.assert_allclose(advector.get_component(1)[:, -1], 0, atol=1e-15)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_just_do_it():
    settings = ColdCumulus()
    settings.kappa = 0
    for process in settings.processes.keys():
        settings.processes[process] = False
    settings.processes['particle advection'] += 1
    settings.processes['fluid advection'] += 1
    simulation = Simulation(settings, Storage(), None)
    simulation.reinit()
    simulation.run(controller=FlowFieldAsserts(simulation))
