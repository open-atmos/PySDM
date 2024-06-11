"""
common logic for environments featuring moist-air thermodynamics
"""

from abc import abstractmethod

import numpy as np


class Moist:
    def __init__(self, dt, mesh, variables, mixed_phase=False):
        variables += ["water_vapour_mixing_ratio", "thd", "T", "p", "RH"]
        if mixed_phase:
            variables += ["a_w_ice"]
        all_vars_unique = len(variables) == len(set(variables))
        assert all_vars_unique

        self.particulator = None
        self.dt = dt
        self.mesh = mesh
        self.variables = variables
        self._values = None
        self._tmp = None
        self._nan_field = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.particulator.observers.append(self)

        if self.particulator.formulae.ventilation.__name__ != "Neglect":
            for var in ("air density", "air dynamic viscosity"):
                if var not in self.variables:
                    self.variables += [var]

        self._values = {"predicted": None, "current": self._allocate(self.variables)}
        self._tmp = self._allocate(self.variables)

        self._nan_field = self._allocate(("_",))["_"]
        self._nan_field.fill(np.nan)

    def _allocate(self, variables):
        result = {}
        for var in variables:
            result[var] = self.particulator.Storage.empty((self.mesh.n_cell,), float)
        return result

    def __getitem__(self, index):
        """returns a Storage representing the variable (field) at a given index or
        otherwise a NaN-filled Storage if the index is not found (in order to simplify
        generic code which uses optional variables, e.g. air viscosity, etc.)"""
        if index in self._values["current"]:
            return self._values["current"][index]
        return self._nan_field

    def get_predicted(self, index):
        if self._values["predicted"] is None:
            raise AssertionError(
                "It seems the AmbientThermodynamics dynamic was not added"
                " when building particulator"
            )
        return self._values["predicted"][index]

    def _recalculate_temperature_pressure_relative_humidity(self, target):
        self.particulator.backend.temperature_pressure_rh(
            rhod=target["rhod"],
            thd=target["thd"],
            water_vapour_mixing_ratio=target["water_vapour_mixing_ratio"],
            T=target["T"],
            p=target["p"],
            RH=target["RH"],
        )

    def sync(self):
        target = self._tmp
        target["water_vapour_mixing_ratio"].ravel(self.get_water_vapour_mixing_ratio())
        target["thd"].ravel(self.get_thd())

        self._recalculate_temperature_pressure_relative_humidity(target)

        if "a_w_ice" in self.variables:
            self.particulator.backend.a_w_ice(
                T=target["T"],
                p=target["p"],
                RH=target["RH"],
                water_vapour_mixing_ratio=target["water_vapour_mixing_ratio"],
                a_w_ice=target["a_w_ice"],
            )
        if "air density" in self.variables:
            self.particulator.backend.air_density(
                water_vapour_mixing_ratio=target["water_vapour_mixing_ratio"],
                rhod=target["rhod"],
                output=target["air density"],
            )
        if "air dynamic viscosity" in self.variables:
            self.particulator.backend.air_dynamic_viscosity(
                temperature=target["T"],
                output=target["air dynamic viscosity"],
            )
        self._values["predicted"] = target

    @abstractmethod
    def get_water_vapour_mixing_ratio(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_thd(self) -> np.ndarray:
        raise NotImplementedError()

    def notify(self):
        if self._values["predicted"] is None:
            return

        self._tmp = self._values["current"]
        self._values["current"] = self._values["predicted"]
        self._values["predicted"] = None
