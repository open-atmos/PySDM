from typing import Union

import matplotlib
import numpy as np
from matplotlib import pyplot
from packaging import version

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.initialisation import discretise_multiplicities
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.physics import constants as const
from PySDM.physics import si
from PySDM.products import IceWaterContent, TotalUnfrozenImmersedSurfaceArea


class Simulation:
    # note: dv and droplet_volume are dummy multipliers (multiplied and then divided by)
    #       will become used if coalescence or other processes are turned on
    def __init__(
        self,
        *,
        cases,
        n_runs_per_case=10,
        multiplicity=1,
        time_step,
        droplet_volume=1 * si.um**3,
        heterogeneous_ice_nucleation_rate="Constant",
        total_time: Union[None, float] = None,
        temperature_range: Union[None, tuple] = None,
    ):
        self.cases = cases
        self.n_runs_per_case = n_runs_per_case
        self.multiplicity = multiplicity
        self.volume = cases.volume
        self.time_step = time_step
        self.droplet_volume = droplet_volume
        self.heterogeneous_ice_nucleation_rate = heterogeneous_ice_nucleation_rate
        self.output = None
        self.total_time = total_time
        self.temperature_range = temperature_range

    def run(self, keys):
        self.output = {}
        for key in keys:
            case = self.cases[key]

            assert (self.total_time is None) + (self.temperature_range is None) == 1
            if self.total_time is not None:
                total_time = self.total_time
            else:
                total_time = (
                    np.diff(np.asarray(self.temperature_range)) / case["cooling_rate"]
                )

            constants = None
            if "J_het" not in case:
                case["J_het"] = None
                constants = {"ABIFM_C": case["ABIFM_c"], "ABIFM_M": case["ABIFM_m"]}
            if "cooling_rate" not in case:
                case["cooling_rate"] = 0
                constants = {"J_HET": case["J_het"]}

            self.output[key] = []
            for i in range(self.n_runs_per_case):
                number_of_real_droplets = case["ISA"].norm_factor * self.volume
                n_sd = number_of_real_droplets / self.multiplicity
                np.testing.assert_approx_equal(n_sd, int(n_sd))
                n_sd = int(n_sd)
                initial_temp = (
                    self.temperature_range[1] if self.temperature_range else np.nan
                )
                f_ufz, a_tot = simulation(
                    constants=constants,
                    seed=i,
                    n_sd=n_sd,
                    time_step=self.time_step,
                    volume=self.volume,
                    spectrum=case["ISA"],
                    droplet_volume=self.droplet_volume,
                    multiplicity=self.multiplicity,
                    total_time=total_time,
                    number_of_real_droplets=number_of_real_droplets,
                    cooling_rate=self.cases[key]["cooling_rate"],
                    heterogeneous_ice_nucleation_rate=self.heterogeneous_ice_nucleation_rate,
                    initial_temperature=initial_temp,
                )
                self.output[key].append({"f_ufz": f_ufz, "A_tot": a_tot})

    def plot(self, ylim, grid=None):
        pyplot.rc("font", size=10)
        for key in self.output:
            for run in range(self.n_runs_per_case):
                time = self.time_step * np.arange(len(self.output[key][run]["f_ufz"]))
                if self.cases[key]["cooling_rate"] == 0:
                    plot_x = time / si.min
                    plot_y = self.output[key][run]["f_ufz"]
                else:
                    plot_x = (
                        self.temperature_range[1]
                        - time * self.cases[key]["cooling_rate"]
                    )
                    plot_y = 1 - np.asarray(self.output[key][run]["f_ufz"])
                pyplot.step(
                    plot_x,
                    plot_y,
                    label=self.cases.label(key) if run == 0 else None,
                    color=self.cases[key]["color"],
                    linewidth=0.666,
                )
        key = None
        if version.parse(matplotlib.__version__) >= version.parse("3.3.0"):
            pyplot.gca().set_box_aspect(1)
        pyplot.legend()
        if grid is not None:
            pyplot.grid(which=grid)
        pyplot.ylim(ylim)
        if self.temperature_range:
            pyplot.xlim(*self.temperature_range)
            pyplot.xlabel("T / K")
            pyplot.ylabel("$f_{frz}$")
        else:
            pyplot.xlim(0, self.total_time / si.min)
            pyplot.xlabel("t / min")
            pyplot.ylabel("$f_{ufz}$")
            pyplot.yscale("log")

    def plot_j_het(self, variant: str, abifm_params_case: str, ylim=None):
        assert variant in ("apparent", "actual")

        formulae = Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            heterogeneous_ice_nucleation_rate="ABIFM",
            constants={
                "ABIFM_M": self.cases[abifm_params_case]["ABIFM_m"],
                "ABIFM_C": self.cases[abifm_params_case]["ABIFM_c"],
            },
        )

        yunit = 1 / si.cm**2 / si.s
        svp = formulae.saturation_vapour_pressure
        plot_x = np.linspace(*self.temperature_range) * si.K
        plot_y = formulae.heterogeneous_ice_nucleation_rate.j_het(
            svp.ice_Celsius(plot_x - const.T0) / svp.pvs_Celsius(plot_x - const.T0)
        )
        pyplot.grid()
        pyplot.plot(plot_x, plot_y / yunit, color="red", label="ABIFM $J_{het}$")

        for key in self.output:
            for run in range(self.n_runs_per_case):
                time = self.time_step * np.arange(len(self.output[key][run]["f_ufz"]))
                if self.cases[key]["cooling_rate"] == 0:
                    raise NotImplementedError()

                temperature = (
                    self.temperature_range[1] - time * self.cases[key]["cooling_rate"]
                )
                spec = self.cases[key]["ISA"]

                particle_number = spec.norm_factor * self.volume
                n_ufz = particle_number * np.asarray(self.output[key][run]["f_ufz"])
                n_frz = particle_number - n_ufz

                j_het = np.diff(n_frz) / self.time_step
                if variant == "apparent":
                    j_het /= n_ufz[:-1] * spec.m_mode
                else:
                    a_tot = np.asarray(self.output[key][run]["A_tot"][:-1])
                    j_het = np.divide(
                        j_het, a_tot, out=np.zeros_like(j_het), where=a_tot != 0
                    )

                pyplot.scatter(
                    temperature[:-1] + np.diff(temperature) / 2,
                    np.where(j_het != 0, j_het, np.nan) / yunit,
                    label=self.cases.label(key) if run == 0 else None,
                    color=self.cases[key]["color"],
                )
        key = None

        pyplot.yscale("log")
        pyplot.xlabel("K")
        pyplot.ylabel(
            f"$J_{{het}}$, $J_{{het}}^{{{variant}}}$ / cm$^{{-2}}$ s$^{{-1}}$"
        )
        pyplot.xlim(self.temperature_range)
        if ylim is not None:
            pyplot.ylim(ylim)
        pyplot.legend()
        if version.parse(matplotlib.__version__) >= version.parse("3.3.0"):
            pyplot.gca().set_box_aspect(1)


def simulation(
    *,
    constants,
    seed,
    n_sd,
    time_step,
    volume,
    spectrum,
    droplet_volume,
    multiplicity,
    total_time,
    number_of_real_droplets,
    cooling_rate=0,
    heterogeneous_ice_nucleation_rate="Constant",
    initial_temperature=np.nan,
):
    formulae = Formulae(
        seed=seed,
        heterogeneous_ice_nucleation_rate=heterogeneous_ice_nucleation_rate,
        constants=constants,
        particle_shape_and_density="MixedPhaseSpheres",
    )
    env = Box(dt=time_step, dv=volume)
    builder = Builder(n_sd=n_sd, backend=CPU(formulae=formulae), environment=env)
    builder.add_dynamic(Freezing(singular=False))

    if hasattr(spectrum, "s_geom") and spectrum.s_geom == 1:
        _isa, _conc = np.full(n_sd, spectrum.m_mode), np.full(
            n_sd, multiplicity / volume
        )
    else:
        _isa, _conc = spectral_sampling.ConstantMultiplicity(spectrum).sample(n_sd)
    attributes = {
        "multiplicity": discretise_multiplicities(_conc * volume),
        "immersed surface area": _isa,
        "volume": np.full(n_sd, droplet_volume),
    }
    np.testing.assert_almost_equal(attributes["multiplicity"], multiplicity)
    products = (
        IceWaterContent(name="qi"),
        TotalUnfrozenImmersedSurfaceArea(name="A_tot"),
    )
    particulator = builder.build(attributes=attributes, products=products)

    env["T"] = initial_temperature
    env["a_w_ice"] = np.nan
    env["RH"] = 1 + np.finfo(float).eps
    svp = particulator.formulae.saturation_vapour_pressure

    cell_id = 0
    f_ufz = []
    a_tot = []
    for i in range(int(total_time / time_step) + 1):
        if cooling_rate != 0:
            env["T"] -= np.full((1,), cooling_rate * time_step / 2)
            env["a_w_ice"] = svp.ice_Celsius(env["T"][0] - const.T0) / svp.pvs_Celsius(
                env["T"][0] - const.T0
            )
        particulator.run(0 if i == 0 else 1)
        if cooling_rate != 0:
            env["T"] -= np.full((1,), cooling_rate * time_step / 2)

        ice_mass_per_volume = particulator.products["qi"].get()[cell_id]
        ice_mass = ice_mass_per_volume * volume
        ice_number = ice_mass / (formulae.constants.rho_w * droplet_volume)
        unfrozen_fraction = 1 - ice_number / number_of_real_droplets
        f_ufz.append(unfrozen_fraction)
        a_tot.append(particulator.products["A_tot"].get()[cell_id])
    return f_ufz, a_tot
