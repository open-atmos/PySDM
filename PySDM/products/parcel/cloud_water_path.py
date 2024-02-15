"""
cloud water path integrated over parcel displacement taking into account changes
in parcel volume along the way
"""

from PySDM.environments.parcel import Parcel

from PySDM.products.impl.activation_filtered_product import _ActivationFilteredProduct
from PySDM.products.impl.moment_product import MomentProduct


class ParcelLiquidWaterPath(MomentProduct, _ActivationFilteredProduct):
    def __init__(
        self,
        count_unactivated: bool,
        count_activated: bool,
        name=None,
        unit="kg/m^2",
    ):
        MomentProduct.__init__(self, unit=unit, name=name)
        _ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )
        self.previous = {"z": 0.0, "cwc": 0.0}
        self.cwp = 0.0

    def register(self, builder):
        if not isinstance(builder.particulator.environment, Parcel):
            raise NotImplementedError()
        _ActivationFilteredProduct.register(self, builder)
        MomentProduct.register(self, builder)
        self.particulator.observers.append(self)

    def notify(self):
        _ActivationFilteredProduct.impl(self, attr="water mass", rank=1)
        avg_mass = self.buffer.copy()

        _ActivationFilteredProduct.impl(self, attr="water mass", rank=0)
        tot_numb = self.buffer.copy()

        self._download_to_buffer(self.particulator.environment["z"])
        current_z = self.buffer.copy()

        cwc = avg_mass * tot_numb / self.particulator.mesh.dv
        dz = current_z - self.previous["z"]
        cwc_mean = (cwc + self.previous["cwc"]) / 2

        if self.previous["cwc"] > 0:
            self.cwp += cwc_mean * dz

        self.previous["z"] = current_z
        self.previous["cwc"] = cwc

    def _impl(self, **kwargs):
        return self.cwp
