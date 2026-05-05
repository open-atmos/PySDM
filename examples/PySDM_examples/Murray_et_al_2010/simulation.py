import numpy as np
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.physics import si


class Simulation:
    def __init__(
        self,
        *,
        cases,
        n_sd=1000,
        n_runs_per_case=1,
        temperature_step=0.01 * si.K,
        droplet_radius=10 * si.um,
        homogeneous_ice_nucleation_rate="KoopMurray2016",
        temperature_range=None,
    ):
        self.cases = cases
        self.n_sd = n_sd
        self.n_runs_per_case = n_runs_per_case
        self.temperature_step = temperature_step
        self.droplet_radius = droplet_radius
        self.homogeneous_ice_nucleation_rate = homogeneous_ice_nucleation_rate
        if temperature_range is None:
            temperature_range = [234.5 * si.K, 238 * si.K]
        self.temperature_range = temperature_range
        self.multiplicity = 1
        self.volume = 1 * si.cm**3
        self.output = None

    def run(self, keys):
        self.output = {}
        for key in keys:
            case = self.cases[key]
            assert case["cooling_rate"] != 0
            total_time = (
                np.diff(np.asarray(self.temperature_range)) / case["cooling_rate"]
            )
            time_step = self.temperature_step / case["cooling_rate"]

            self.output[key] = []
            for i in range(self.n_runs_per_case):
                T_frz = simulation(
                    seed=i,
                    n_sd=self.n_sd,
                    time_step=time_step,
                    volume=self.volume,
                    droplet_radius=self.droplet_radius,
                    cooling_rate=case["cooling_rate"],
                    multiplicity=self.multiplicity,
                    total_time=total_time,
                    homogeneous_ice_nucleation_rate=self.homogeneous_ice_nucleation_rate,
                    initial_temperature=self.temperature_range[1],
                )
                self.output[key].append({"T_frz": T_frz})

    def plot_histogram(self, title=None):

        pyplot.rc("font", size=12)
        T_bins = np.arange(self.temperature_range[0], self.temperature_range[1], 0.1)

        for key in self.output:
            for run in range(self.n_runs_per_case):
                if run == 0:
                    cooling_rate = self.cases[key]["cooling_rate"]
                    label = (
                        "-"
                        + f"{cooling_rate*si.minute:.1f}"
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

        if title is None:
            title = (
                r"$r_\mathrm{drop}:$ "
                + f"{self.droplet_radius/si.micrometer:.2f}"
                + "µm"
                + r"   $\mathrm{N_{sd}}:$ "
                + str(self.n_sd)
                + "   dT: "
                + f"{self.temperature_step:.2f}"
                + "K"
            )
        pyplot.title(title, pad=15)
        pyplot.xlim(*self.temperature_range)
        pyplot.ylim(-0.05, 1.05)
        pyplot.xlabel("Temperature / K")
        pyplot.ylabel("Fraction of droplets frozen")
        pyplot.minorticks_on()
        pyplot.tick_params(axis="both", which="both", top=True, right=True)
        pyplot.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
        pyplot.tick_params(which="minor", length=3)
        pyplot.tick_params(which="major", length=6)
        pyplot.legend()


def simulation(
    *,
    seed,
    n_sd,
    time_step,
    volume,
    droplet_radius,
    cooling_rate,
    multiplicity,
    total_time,
    homogeneous_ice_nucleation_rate="KoopMurray2016",
    initial_temperature=None,
):
    formulae = Formulae(
        seed=seed,
        homogeneous_ice_nucleation_rate=homogeneous_ice_nucleation_rate,
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
    droplet_volume = formulae.trivia.volume(radius=droplet_radius)

    attributes = {
        "multiplicity": np.full(n_sd, multiplicity),
        "signed water mass": np.full(n_sd, droplet_volume * formulae.constants.rho_w),
    }
    particulator = builder.build(attributes=attributes)
    env = particulator.environment

    env["T"] = initial_temperature
    env["a_w_ice"] = np.nan
    env["RH"] = 1 + np.finfo(float).eps
    svp = particulator.formulae.saturation_vapour_pressure

    cell_id = 0
    for i in range(int(total_time / time_step) + 1):
        env["RH_ice"] = svp.pvs_water(env["T"][cell_id]) / svp.pvs_ice(
            env["T"][cell_id]
        )
        env["a_w_ice"] = svp.pvs_ice(env["T"][cell_id]) / svp.pvs_water(
            env["T"][cell_id]
        )
        env["T"] -= np.full((1,), cooling_rate * time_step)
        particulator.run(0 if i == 0 else 1)

    assert all(particulator.attributes["signed water mass"].data < 0)
    T_frz = particulator.attributes["temperature of last freezing"].data.tolist()
    return T_frz
