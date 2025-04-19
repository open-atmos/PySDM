
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle
import json

from PySDM.physics.constants import si

from settings import settings as simulation_settings
from simulation import Simulation
# from plot import plot_size_distribution, plot_evolution, plot_ensemble

general_settings = {"n_sd": 10, "T0": 216 * si.kelvin,
                    "w_updraft": 10 * si.centimetre / si.second}
distributions = ({"N_dv_solution_droplet": 2500 / si.centimetre ** 3,
                  "r_mean_solution_droplet": 0.055 * si.micrometre,
                  "sigma_solution_droplet": 1.6},
                 {"N_dv_solution_droplet": 8600 / si.centimetre**3,
                  "r_mean_solution_droplet": 0.0275 * si.micrometre,
                  "sigma_solution_droplet": 1.3},
                 {"N_dv_solution_droplet": 2000 / si.centimetre**3,
                  "r_mean_solution_droplet": 0.11 * si.micrometre,
                  "sigma_solution_droplet": 2.},
                 )

# pp = PdfPages("hom_freezing_for_size_distributions.pdf")
#
# for distribution in distributions:
#     setting = settings(**{**general_settings, **distribution})
#     model = Simulation(setting)
#     output = model.run()
#
#     plot_size_distribution(model.r_wet, model.r_dry, model.multiplicities, setting, pp)
#     plot(output, setting, model, pp)



# calculate super particle ensemble
def ensemble_simulation(number_of_ensemble_runs=1,
                        dsd=0,
                        T0=216.* si.kelvin,
                        w_updraft=None,
                        linear_sampling=False,
                        nsd_single = None,
                        lower_limit = None,
                        add_label = "",
                        RHi_0 = None,
                        ):

    file_name = ("ensemble_"+str(number_of_ensemble_runs)+"_dsd_"+str(dsd)
                 + f"_T0_{T0:.0f}" +  f"_W_{w_updraft:.2f}")


    if nsd_single is None:
        number_of_super_particles = (50, 100, 500, 1000, 5000, 10000, 50000, 100000)
    else:
        number_of_super_particles  = (nsd_single,)
        file_name += "_nsd_"+str(nsd_single)

    if linear_sampling:
        file_name += "_lin"

    if lower_limit is not None:
        file_name += "_limit"
    file_name += add_label


    outputs = []

    aerosol_distribution = distributions[dsd]
    setting = {**general_settings,
               **aerosol_distribution,
               "linear_sampling": linear_sampling,
               "lower_limit": lower_limit}

    if T0 is not None:
        setting["T0"] = T0
    if RHi_0 is not None:
        setting["RHi_0"] = RHi_0
    if w_updraft is not None:
        setting["w_updraft"] = w_updraft

    for nsd in number_of_super_particles:
        setting["n_sd"] = nsd

        print(setting)

        for _ in range(number_of_ensemble_runs):
            simulation_setting = simulation_settings(**setting)
            model = Simulation(simulation_setting)
            output = model.run()
            outputs.append(output)
            del model, simulation_setting

    data_file = { "outputs":outputs,
                  "number_of_ensemble_runs": number_of_ensemble_runs,
                  "initial_temperature": setting["T0"],
                  "aerosol_distribution": aerosol_distribution,
                  "w_updraft": setting["w_updraft"],
                  }
    print("Writing "+file_name+".json")
    with open(file_name+".json", 'w') as file:
        json.dump(data_file, file)


# ensemble_simulation(1)
# setting = {**general_settings, **distributions[0]  }
# simulation_setting = simulation_settings(**setting)
#
# setting = {**general_settings, **distributions[0], "linear_sampling": True  }
# simulation_setting = simulation_settings(**setting)


# for DSD plots
# ensemble_simulation(nsd_single = 50000, dsd=0, add_label="koop_corr", w_updraft=1., number_of_ensemble_runs=25)
# ensemble_simulation(nsd_single = 50, dsd=0, T0=220* si.kelvin,
#                     RHi_0=1.6,
#                     add_label="_RH16")
# ensemble_simulation(nsd_single = 50, dsd=0, T0=220* si.kelvin,
#                     RHi_0=1.0,
#                     add_label="_RH10")
# ensemble_simulation(nsd_single = 50, dsd=0, T0=220* si.kelvin,
#                     RHi_0=1.4,
#                     add_label="_RH14")
# ensemble_simulation(nsd_single = 50, dsd=0, T0=220* si.kelvin,
#                     RHi_0=1.4,)
# ensemble_simulation(nsd_single = 50, dsd=0, T0=200* si.kelvin,
#                     RHi_0=1.4,)
# ensemble_simulation(nsd_single = 50, dsd=0, T0=190* si.kelvin,
#                     RHi_0=1.4,)
# ensemble_simulation(nsd_single = 50, dsd=1)
# ensemble_simulation(nsd_single = 50, dsd=2)
# ensemble_simulation(nsd_single = 50, dsd=0, lower_limit=5.5e-8)
# ensemble_simulation(nsd_single = 50, dsd=0, linear_sampling=True)

# ensemble_simulation(nsd_single = 50000, dsd=1,
#                     RHi_0=1.0, w_updraft=1.*si.meter / si.second,
#                     add_label="highoutput")


# ensemble_simulation(number_of_ensemble_runs=25, w_updraft=1.)
# ensemble_simulation( linear_sampling=True, number_of_ensemble_runs=25, w_updraft=1.)
#
# lower_limit_bound = (  5.5e-8, )
# for lower_limit in lower_limit_bound:
#     ensemble_simulation( lower_limit=lower_limit, number_of_ensemble_runs=25, w_updraft=1.)

initial_temperatures = [196.,216.,236.]
updrafts = [0.05, 0.1, 0.5, 1., 5., 10.]
number_of_ensemble_runs=1
dsd=0

for T in reversed(initial_temperatures):
    for w in reversed(updrafts):
        ensemble_simulation(number_of_ensemble_runs=number_of_ensemble_runs,
                            w_updraft=w,
                            T0=T,
                            dsd=dsd,
                            linear_sampling=False,
                            nsd_single = 10,
                            )