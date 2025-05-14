import json
from examples.PySDM_examples.Spichtinger_et_al_2023 import Simulation, Settings
from examples.PySDM_examples.Spichtinger_et_al_2023.data import simulation_data
import numpy as np
from matplotlib import pyplot


calculate_data = False
save_to_file   = False
plot           = True
read_from_json = False


if calculate_data:

    general_settings = {"n_sd": 50000, "dt": 0.1}

    initial_temperatures = np.array([196., 216., 236.])
    updrafts = np.array([0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1., 3., 5.])
    number_of_ensemble_runs = 5
    seeds = [124670285330, 439785398735, 9782539783258, 12874192127481, 12741731272]

    dim_updrafts = len(updrafts)
    dim_initial_temperatures = len(initial_temperatures)

    number_concentration_ice = np.zeros([dim_updrafts, dim_initial_temperatures, number_of_ensemble_runs])

    for i in range(dim_updrafts):
        for j in range(dim_initial_temperatures):
            for k in range(number_of_ensemble_runs):
                setting_dictionary = {**general_settings, }
                setting_dictionary["w_updraft"] = updrafts[i]
                setting_dictionary["T0"] = initial_temperatures[j]
                setting_dictionary["seed"] = seeds[k]
                setting = Settings(**setting_dictionary)
                model = Simulation(setting)
                number_concentration_ice[i,j,k] = model.run()

    if save_to_file:
        file_name = "data/ni_w_T_ens_"+str(number_of_ensemble_runs)+".json"
        data_file  = {"ni": number_concentration_ice.tolist(),
                      "T": initial_temperatures.tolist(),
                      "w": updrafts.tolist()}
        with open(file_name, 'w') as file:
            json.dump(data_file, file)

if plot:
    if calculate_data:
        T = initial_temperatures
        w = updrafts
        ni_ens_mean = np.mean(number_concentration_ice, axis=2)
    else:
        if read_from_json:
            file_name = "data/ni_w_T_ens_5.json"
            with open(file_name, 'r') as f:
                data = json.load(f)

            ni = data["ni"]
            T = data["T"]
            w = data["w"]
            ni_ens_mean = np.mean(ni, axis=2)
        else:
            T, w, ni_ens_mean = simulation_data.saved_simulation_ensemble_mean()


    # plot
    fig, ax = pyplot.subplots(1, 1, figsize=(5, 5))

    for j in range(len(T)):
        ax.scatter(w, ni_ens_mean[:, j], label=f"T0={T[j]:.0f}K")

    ax.set_xscale('log')
    ax.set_xlim(0.08, 10.)
    ax.set_xlabel(r"vertical updraft [$\mathrm{m \, s^{-1}}$]")

    ax.set_yscale('log')
    ax.set_ylim(1.e2, 1.e10)
    ax.set_ylabel(r"ice number concentration [$\mathrm{cm^{-3}}$]")

    ax.legend(loc="lower right")

    pyplot.show()
