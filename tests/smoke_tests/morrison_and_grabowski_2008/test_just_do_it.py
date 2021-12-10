# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from contextlib import AbstractContextManager
import numpy as np
from PySDM_examples.Szumowski_et_al_1998 import Simulation, Storage
from PySDM_examples.Morrison_and_Grabowski_2008 import ColdCumulus


class FlowFieldAsserts(AbstractContextManager):
    def __init__(self, simulation):
        self.particulator = simulation.particulator
        self.panic = None

    def set_percent(self, percent):
        if percent == 0:
            return
        advector = None
        solvers = self.particulator.dynamics['EulerianAdvection'].solvers.mpdatas.values()
        for solver in solvers:
            assert advector is None or advector is solver.advector
            advector = solver.advector
        np.testing.assert_allclose(advector.get_component(1)[:, 0], 0)
        np.testing.assert_allclose(advector.get_component(1)[:, -1], 0, atol=1e-15)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_just_do_it():
    settings = ColdCumulus()
    settings.kappa = 0
    for process in settings.processes:
        settings.processes[process] = False
    settings.processes['particle advection'] += 1
    settings.processes['fluid advection'] += 1
    simulation = Simulation(settings, Storage(), None)
    simulation.reinit()
    simulation.run(controller=FlowFieldAsserts(simulation))
