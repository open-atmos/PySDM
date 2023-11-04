from PySDM_examples.Alpert_and_Knopf_2016.table import Table

from PySDM.initialisation.spectra import Lognormal, TopHat
from PySDM.physics import si


class Table1(Table):
    def label(self, key):
        if isinstance(self[key]["ISA"], Lognormal):
            return (
                f"σ=ln({int(self[key]['ISA'].s_geom)}),"
                f"N={int(self[key]['ISA'].norm_factor * self.volume)}"
            )
        return key

    def __init__(self, *, volume=1 * si.cm**3):
        super().__init__(
            volume=volume,
            data={
                "Iso1": {
                    "ISA": Lognormal(
                        norm_factor=1000 / volume, m_mode=1e-5 * si.cm**2, s_geom=1
                    ),
                    "color": "#298131",
                    "J_het": 1e3 / si.cm**2 / si.s,
                },
                "Iso2": {
                    "ISA": Lognormal(
                        norm_factor=30 / volume, m_mode=1e-5 * si.cm**2, s_geom=1
                    ),
                    "color": "#9ACFA4",
                    "J_het": 1e3 / si.cm**2 / si.s,
                },
                "Iso3": {
                    "ISA": Lognormal(
                        norm_factor=1000 / volume, m_mode=1e-5 * si.cm**2, s_geom=10
                    ),
                    "color": "#1A62B4",
                    "J_het": 1e3 / si.cm**2 / si.s,
                },
                "Iso4": {
                    "ISA": Lognormal(
                        norm_factor=30 / volume, m_mode=1e-5 * si.cm**2, s_geom=10
                    ),
                    "color": "#95BDE1",
                    "J_het": 1e3 / si.cm**2 / si.s,
                },
                "IsoWR": {
                    "ISA": Lognormal(
                        norm_factor=1000 / volume,
                        m_mode=6.4e-3 * si.cm**2,
                        s_geom=9.5,
                    ),
                    "color": "#FED2B0",
                    "J_het": 6e-4 / si.cm**2 / si.s,
                },
                "IsoBR": {
                    "ISA": TopHat(
                        norm_factor=63 / volume,
                        endpoints=(9.4e-8 * si.cm**2, 7.5e-7 * si.cm**2),
                    ),
                    "color": "#FED2B0",
                    "J_het": 2.8e3 / si.cm**2 / si.s,
                },
                "IsoHE1": {
                    "ISA": Lognormal(
                        norm_factor=40 / volume, m_mode=1.2 * si.cm**2, s_geom=2.2
                    ),
                    "color": "#FED2B0",
                    "J_het": 4.1e-3 / si.cm**2 / si.s,
                },
                "IsoHE2": {
                    "ISA": Lognormal(
                        norm_factor=40 / volume, m_mode=2e-2 * si.cm**2, s_geom=8.5
                    ),
                    "color": "#FED2B0",
                    "J_het": 2e-2 / si.cm**2 / si.s,
                },
                "IsoDI1": {
                    "ISA": Lognormal(
                        norm_factor=45 / volume, m_mode=5.1e-1 * si.cm**2, s_geom=3.2
                    ),
                    "J_het": 1.8e-2 / si.cm**2 / si.s,
                    "color": "#9ACFA4",
                },
                "IsoDI2": {
                    "ISA": Lognormal(
                        norm_factor=45 / volume, m_mode=5.1e-2 * si.cm**2, s_geom=3.2
                    ),
                    "J_het": 1 / si.cm**2 / si.s,
                    "color": "#FED2B0",
                },
                "IsoDI3": {
                    "ISA": Lognormal(
                        norm_factor=45 / volume, m_mode=5.1e-1 * si.cm**2, s_geom=3.2
                    ),
                    "J_het": 1 / si.cm**2 / si.s,
                    "color": "#95BDE1",
                },
            },
        )
