import os
import numpy as np
import json

from matplotlib import pyplot as plt
from PySDM_examples.eware_2024.example import run,Settings,SpectrumPlotter
# from PySDM_examples.Shima_et_al_2009.settings import Settings
# from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumPlotter
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity,Logarithmic,Linear
from PySDM.backends import CPU


def main(plot: bool = True, save: str = None):
    backend = CPU()
    n_sds = [12,13,14,15,16,17,18,19]
    dts = [20,10,5,2,1]#, "adaptive"]
    sampling_strat = [ConstantMultiplicity]
    sampling_strat_names = ["ConstantMultiplicity"]
    regular = {"ConstantMultiplicity":{}}
    # adaptive = {"ConstantMultiplicity":{}}
    iters_without_warmup = 5
    base_time = None
    base_error = None

    error_heatmaps = {}
    error_std_heatmaps = {}
    deficit_heatmaps = {}

    for k,strat in enumerate(sampling_strat):
        error_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        error_std_hm = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        deficit_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        mean_time_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        sanity_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        plotter = SpectrumPlotter(Settings(), legend=False)
        plotter.smooth = False
        for i, dt in enumerate(dts):
            for j, n_sd in enumerate(n_sds):
                sanity_heatmap[i][j] = "dt="+str(dt)+", n_sd="+str(n_sd)
                outputs = []
                deficits = []
                errors = []
                exec_times = []
                one_for_warmup = 1
                for it in range(iters_without_warmup + one_for_warmup):
                    settings = Settings()
                    backend.formulae.seed = it

                    settings.n_sd = 2**n_sd
                    settings.dt = dt #if dt != "adaptive" else max(dts[:-1])
                    settings.adaptive = False #dt == "adaptive"
                    settings.sampling = strat(settings.spectrum)

                    states, exec_time, deficit = run(settings,backend)
                    deficit *= settings.dv
                    print(f"{dt=}, {n_sd=}, {exec_time=}, {it=}")
                    if it != 0:
                        exec_times.append(exec_time)
                        outputs.append(states)
                        deficits.append(deficit)

                        for step, vals in states.items():
                            error = plotter.plot(vals, step * settings.dt)
                        errors.append(error*1e-3) #grams to kg
                mean_time = np.mean(exec_times)#[one_for_warmup:])
                if base_time is None:
                    base_time = mean_time
                norm_time = mean_time / base_time
                mean_output = {}
                # mean_deficit = {}
                for key in outputs[0].keys():
                    mean_output[key] = sum((output[key] for output in outputs)) / len(
                        outputs
                    )
                # for key in deficits[0].keys():
                #     mean_deficit[key] = sum((deficit[key] for deficit in deficits)) / len(
                #         deficits
                #     )
                mean_deficit = sum(deficits) / len(deficits)
                last_step_error = sum(errors) / len(errors)
                error_std = np.std(errors)

                # for step, vals in mean_output.items():
                #     error = plotter.plot(vals, step * settings.dt)
                # last_step_error = error*1e-3 #grams to kg
                # if base_error is None:
                #     base_error = last_step_error
                # norm_error = last_step_error / base_error

                plotter.ylabel = (
                    r"$\bf{dt: "
                    + str(settings.dt)
                    + ("+ adaptivity" if settings.adaptive else "")
                    + "}$\ndm/dlnr [g/m^3/(unit dr/r)]"
                    if j == 0
                    else None
                )
                plotter.xlabel = (
                    "particle radius [µm]\n" + r"$\bf{n_{sd}: 2^{" + str(n_sd) + "}}$"
                    if i == len(dts) - 1
                    else None
                )
                plotter.title = (
                    f"time: {norm_time:.2f} error: {last_step_error:.2f} (normalised)"
                )
                error_heatmap[i][j] = last_step_error
                error_std_hm[i][j] = error_std
                deficit_heatmap[i][j] = mean_deficit #sum(mean_deficit)/3600
                mean_time_heatmap[i][j] = mean_time
                plotter.finished = False
                plotter.finish()
        regular[sampling_strat_names[k]]["Error"] = error_heatmap
        regular[sampling_strat_names[k]]["Error_std"] = error_std_hm
        regular[sampling_strat_names[k]]["Deficit"] = deficit_heatmap
        regular[sampling_strat_names[k]]["MeanTime"] = mean_time_heatmap

    print(sanity_heatmap)
    # if save is not None:
    #     n_sd = settings.n_sd
    #     plotter.save(save + "/" + f"{n_sd}_shima_fig_2" + "." + plotter.format)

    if plot:
        plotter.show()
        plt.figure()
        for k,strat in enumerate(sampling_strat):
            print(strat,error_heatmaps[sampling_strat_names[k]])
            plt.imshow(error_heatmaps[sampling_strat_names[k]], cmap='viridis')
        plt.show()
        plt.figure()
        for k,strat in enumerate(sampling_strat):
            print(strat,deficit_heatmaps[sampling_strat_names[k]])
            plt.imshow(deficit_heatmaps[sampling_strat_names[k]], cmap='viridis')
        plt.show()

    return regular,sanity_heatmap


if __name__ == "__main__":
    regular,type_matrix = main(plot=False, save=".")
    
results = {"regular":regular,"type_matrix":type_matrix}
# with open('test_runs_6_29.json', 'w', encoding='UTF-8') as f:
#     json.dump(results, f)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

error_matrix = np.array(regular["ConstantMultiplicity"]["Error"])
deficit_matrix = np.array(regular["ConstantMultiplicity"]["Deficit"])

deficit_matrix_flipped = np.flipud(deficit_matrix)
deficit_matrix_flipped = deficit_matrix_flipped[:,:,0]
#replace 0s with np.nan
deficit_matrix_flipped[deficit_matrix_flipped == 0] = np.nan

# Create the figure and axes
fig, axes = plt.subplots(2, 1, figsize=(4, 6))

# Define the grid for contours
runs = [12, 13, 14, 15, 16, 17, 18, 19]
dts = [20, 10, 5, 2, 1]
X, Y = np.meshgrid(np.arange(len(runs)), np.arange(len(dts)))

contour = axes[0].contourf(
    X, Y, error_matrix, levels=20, cmap="mako")
contour_lines = axes[0].contour(
    X, Y, error_matrix, levels=20, colors="lightblue", linewidths=0.8,
)
#xticks and y-ticks

axes[0].set_yticks(np.arange(len(dts)))
axes[0].set_yticklabels(dts)

cbar = fig.colorbar(contour, ax=axes[0], orientation="vertical", label=f"RMSE ($kg/m^3$)")
cbar.add_lines(contour_lines)  # Add contour lines to the colorbar
axes[0].set_title("RMSE at t = 3600s")
axes[0].set_xlabel(f"$N_s$")
axes[0].set_ylabel("dt(s)")

# Plot the deficit heatmap
sns.heatmap(deficit_matrix_flipped,
    ax=axes[1],cmap="mako_r",cbar=True,
    cbar_kws={"label": f"Deficit ($collisions\ s^{{-1}} m^{{-3}}$)"},
)
axes[0].set_xticks(np.arange(len(runs)))
axes[0].set_xticklabels([f"$2^{{{n}}}$" for n in runs])  # Format as 2^n_sd
axes[1].set_xticks(np.arange(len(runs)) + 0.5)
axes[1].set_xticklabels([f"$2^{{{n}}}$" for n in runs])  # Format as 2^n_sd

axes[1].set_yticks(np.arange(len(dts)) + 0.5)
axes[1].set_yticklabels(dts[::-1])  # Reverse the order of

axes[1].set_title("Deficit")
axes[1].set_xlabel(f"$N_s$")
axes[1].set_ylabel("dt(s)")

plt.tight_layout()
plt.savefig("Shima_2009_deficit.pdf")
plt.show()
# %%
