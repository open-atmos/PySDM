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
from PySDM.physics import si
from PySDM.products import IceWaterContent


class Simulation:
    def __init__(
        self,
        *,
        cases,
        n_sd=100,
        n_runs_per_case=1,
        # multiplicity=1,
        time_step=1 * si.s,
        droplet_radius=10 * si.um,
        # cooling_rate=-5 * si.K / si.minute,
        droplet_concentration=1 * si.micrometer**3.0,
        # initial_temperature= 238 * si.K,
        homogeneous_ice_nucleation_rate="KoopMurray2016",
        # total_time: Union[None, float] = None,
        temperature_range=[234.5 * si.K, 238 * si.K],
    ):
        self.n_sd = n_sd
        self.cases = cases
        self.n_runs_per_case = n_runs_per_case
        self.multiplicity = 1
        self.volume = 1 * si.cm**3
        self.time_step = time_step
        self.droplet_radius = droplet_radius
        self.droplet_concentration = droplet_concentration
        # self.cooling_rate = cooling_rate
        # self.initial_temperature = initial_temperature
        self.homogeneous_ice_nucleation_rate = homogeneous_ice_nucleation_rate
        self.output = None
        # self.total_time = total_time
        self.temperature_range = temperature_range

    def run(self, keys):
        self.output = {}
        for key in keys:
            case = self.cases[key]

            # print(  self.total_time, self.temperature_range  )
            # assert (self.total_time is None) + (self.temperature_range is None) == 1
            # if self.total_time is not None:
            #     total_time = self.total_time
            # else:
            total_time = (
                np.diff(np.asarray(self.temperature_range)) / case["cooling_rate"]
            )
            print(
                self.temperature_range,
                np.diff(np.asarray(self.temperature_range)),
                total_time,
            )
            constants = None
            # if "J_het" not in case:
            #     case["J_het"] = None
            #     constants = {"ABIFM_C": case["ABIFM_c"], "ABIFM_M": case["ABIFM_m"]}
            # if "cooling_rate" not in case:
            #     case["cooling_rate"] = 0
            #     constants = {"J_HET": case["J_het"]}

            self.output[key] = []
            for i in range(self.n_runs_per_case):
                # number_of_real_droplets = self.droplet_concentration * self.volume
                # n_sd = number_of_real_droplets / self.multiplicity
                # np.testing.assert_approx_equal(n_sd, int(n_sd))
                # n_sd = int(n_sd)
                # initial_temp = (
                #     self.temperature_range[1] if self.temperature_range else np.nan
                # )
                f_ufz, T_frz = simulation(
                    constants=constants,
                    seed=i,
                    n_sd=self.n_sd,
                    time_step=self.time_step,
                    volume=self.volume,
                    # spectrum=case["ISA"],
                    droplet_radius=self.droplet_radius,
                    cooling_rate=case["cooling_rate"],
                    multiplicity=self.multiplicity,
                    total_time=total_time,
                    # number_of_real_droplets=number_of_real_droplets,
                    homogeneous_ice_nucleation_rate=self.homogeneous_ice_nucleation_rate,
                    initial_temperature=self.temperature_range[1],
                )
                self.output[key].append({"f_ufz": f_ufz, "T_frz": T_frz})

    def plot_histogram(self):

        pyplot.rc("font", size=10)
        T_bins = np.arange(self.temperature_range[0], self.temperature_range[1], 0.1)

        for key in self.output:
            for run in range(self.n_runs_per_case):

                if run == 0:
                    cooling_rate = self.cases[key]["cooling_rate"]
                    label = (
                        "-"
                        + f"{cooling_rate * si.minute:.1f}"
                        + r" $\mathrm{K \, min^{-1}}$"
                    )
                else:
                    label = None

                pyplot.hist(
                    self.output[key][run]["T_frz"],
                    bins=T_bins,
                    cumulative=-1,
                    density=True,
                    alpha=1.0,
                    histtype="step",
                    linewidth=1.5,
                    color=self.cases[key]["color"],
                    label=label,
                )
        pyplot.xlim(*self.temperature_range)
        pyplot.xlabel("Temperature / K")
        pyplot.ylabel("Fraction of droplets froze")
        pyplot.legend()


def simulation(
    *,
    constants,
    seed,
    n_sd,
    time_step,
    volume,
    # spectrum,
    droplet_radius,
    cooling_rate,
    multiplicity,
    total_time,
    # number_of_real_droplets,
    homogeneous_ice_nucleation_rate="KoopMurray2016",
    initial_temperature=None,
):
    formulae = Formulae(
        seed=seed,
        homogeneous_ice_nucleation_rate=homogeneous_ice_nucleation_rate,
        constants=constants,
        particle_shape_and_density="MixedPhaseSpheres",
        saturation_vapour_pressure="MurphyKoop2005",
    )
    builder = Builder(
        n_sd=n_sd,
        backend=CPU(formulae=formulae),
        environment=Box(dt=time_step, dv=volume),
    )
    builder.add_dynamic(
        Freezing(homogeneous_freezing="time-dependent", immersion_freezing=None)
    )
    builder.request_attribute("temperature of last freezing")
    builder.request_attribute("volume")
    builder.request_attribute("radius")

    droplet_volume = formulae.trivia.volume(radius=droplet_radius)
    print(droplet_radius, droplet_volume)

    attributes = {
        "multiplicity": np.full(n_sd, multiplicity),
        "signed water mass": np.full(n_sd, droplet_volume * formulae.constants.rho_w),
    }
    # np.testing.assert_almost_equal(attributes["multiplicity"], multiplicity)
    products = (
        IceWaterContent(name="qi"),
        # TotalUnfrozenImmersedSurfaceArea(name="A_tot"),
    )
    particulator = builder.build(attributes=attributes, products=products)
    env = particulator.environment

    env["T"] = initial_temperature
    env["a_w_ice"] = np.nan
    env["RH"] = 1 + np.finfo(float).eps
    svp = particulator.formulae.saturation_vapour_pressure
    # env["RH_ice"] = svp.pvs_water(env["T"][0]) / svp.pvs_ice(env["T"][0])
    # print(env["RH_ice"][0])

    number_of_real_droplets = n_sd * multiplicity
    cell_id = 0
    f_ufz = []
    # a_tot = []
    print(int(total_time / time_step) + 1)
    for i in range(int(total_time / time_step) + 1):
        env["RH_ice"] = svp.pvs_water(env["T"][0]) / svp.pvs_ice(env["T"][0])
        env["a_w_ice"] = svp.pvs_ice(env["T"][0]) / svp.pvs_water(env["T"][0])
        # env["a_w_ice"]= 1. / env["RH_ice"]
        if cooling_rate != 0:
            print(env["T"][0], cooling_rate * time_step)
            env["T"] -= np.full((1,), cooling_rate * time_step)
        particulator.run(0 if i == 0 else 1)

        ice_mass_per_volume = particulator.products["qi"].get()[cell_id]
        ice_mass = ice_mass_per_volume * volume
        ice_number = ice_mass / (formulae.constants.rho_w * droplet_volume)
        unfrozen_fraction = 1 - ice_number / number_of_real_droplets
        f_ufz.append(unfrozen_fraction)
        # print(i, env["T"][0], env["RH_ice"][0], env["a_w_ice"][0], unfrozen_fraction)
    assert all(particulator.attributes["signed water mass"].data < 0)
    T_frz = particulator.attributes["temperature of last freezing"].data.tolist()
    return f_ufz, T_frz
