from importlib import reload
from PySDM.simulation.physics import constants, _fake_unit_registry


class DimensionalAnalysis:
    def __enter__(*_):
        _fake_unit_registry.FAKE_UNITS = False
        reload(constants)

    def __exit__(*_):
        _fake_unit_registry.FAKE_UNITS = True
        reload(constants)
