from PySDM_examples.Alpert_and_Knopf_2016.table import Table

from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si


class Table2(Table):
    def label(self, key):
        return f"r={self[key]['cooling_rate']/(si.K/si.min)} K/min"

    def __init__(self, *, volume=1 * si.cm**3):
        super().__init__(
            volume=volume,
            data={
                "Cr1": {
                    "ISA": Lognormal(
                        norm_factor=1000 / volume, s_geom=10, m_mode=1e-5 * si.cm**2
                    ),
                    "cooling_rate": 0.5 * si.K / si.min,
                    "color": "orange",
                    "ABIFM_c": -10.67,
                    "ABIFM_m": 54.48,
                },
                "Cr2": {
                    "ISA": Lognormal(
                        norm_factor=1000 / volume, s_geom=10, m_mode=1e-5 * si.cm**2
                    ),
                    "cooling_rate": 5 * si.K / si.min,
                    "color": "blue",
                    "ABIFM_c": -10.67,
                    "ABIFM_m": 54.48,
                },
                "CrHE1": {
                    "ISA": Lognormal(
                        norm_factor=40 / volume, s_geom=8.5, m_mode=2.1e-2 * si.cm**2
                    ),
                    "cooling_rate": 0.2 * si.K / si.min,
                    "color": "orange",
                    "ABIFM_c": -12.98,
                    "ABIFM_m": 122.83,
                },
                "CrHE2": {
                    "ISA": Lognormal(
                        norm_factor=40 / volume, s_geom=8.5, m_mode=2.1e-2 * si.cm**2
                    ),
                    "cooling_rate": 2 * si.K / si.min,
                    "color": "blue",
                    "ABIFM_c": -12.98,
                    "ABIFM_m": 122.83,
                },
            },
        )
