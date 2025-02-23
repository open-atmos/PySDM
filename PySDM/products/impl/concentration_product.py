"""
common code for products computing particle concentrations with
the option to return them at standard temperature and pressure (STP) conditions
"""

from PySDM.products.impl.moment_product import MomentProduct


class ConcentrationProduct(MomentProduct):
    @staticmethod
    def check_ctor_arguments(specific, stp):
        if stp and specific:
            raise ValueError(
                "std-temperature-and-pressure precludes specific conc. option"
            )

    def __init__(self, *, unit: str, name: str, specific: bool, stp: bool):
        """
        `stp` toggles expressing the concentration in terms of standard temperature
        and pressure conditions (ground level of the ICAO standard atmosphere, zero humidity)
        """
        self.check_ctor_arguments(specific, stp)
        super().__init__(unit=unit, name=name)
        self.specific = specific
        self.stp = stp
        self.rho_stp = None

    def register(self, builder):
        super().register(builder)
        self.rho_stp = builder.formulae.constants.rho_STP

    def _impl(self, **kwargs):
        assert len(kwargs) == 0

        self.buffer[:] /= self.particulator.mesh.dv

        if self.specific or self.stp:
            result = self.buffer.copy()
            self._download_to_buffer(self.particulator.environment["rhod"])
            result[:] /= self.buffer
            if self.stp:
                result[:] *= self.rho_stp
        else:
            result = self.buffer

        return result
