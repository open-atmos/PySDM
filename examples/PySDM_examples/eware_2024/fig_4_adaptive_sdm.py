import os
import numpy as np
import json

from matplotlib import pyplot as plt
from PySDM_examples.eware_2024.example import run,Settings,SpectrumPlotter
# from PySDM_examples.Shima_et_al_2009.settings import Settings
# from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumPlotter
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity,Logarithmic,Linear


def main(plot: bool = True, save: str = None):
    n_sds = [13,14,15,16,17,18,19]
    dts = [20,10,5]#, "adaptive"]
    sampling_strat = [ConstantMultiplicity,Logarithmic, Linear]
    sampling_strat_names = ["ConstantMultiplicity","Logarithmic", "Linear"]
    regular = {"ConstantMultiplicity":{},"Logarithmic":{}, "Linear":{}}
    adaptive = {"ConstantMultiplicity":{},"Logarithmic":{}, "Linear":{}}
    iters_without_warmup = 1
    base_time = None
    base_error = None

    plt.ioff()
    fig, axs = plt.subplots(
        len(dts), len(n_sds), sharex=True, sharey=True, figsize=(10, 10)
    )

    error_heatmaps = {}
    error_std_heatmaps = {}
    deficit_heatmaps = {}
    mean_time_heatmaps = {}
    error_heatmaps_adaptive = {}
    error_std_heatmaps_adaptive = {}
    deficit_heatmaps_adaptive = {}
    mean_time_heatmaps_adaptive = {}



    for k,strat in enumerate(sampling_strat):
        error_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        error_std_hm = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        deficit_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        mean_time_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        sanity_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        plotter = SpectrumPlotter(Settings(seed=42), legend=False)
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
                    settings = Settings(seed=it)

                    settings.n_sd = 2**n_sd
                    settings.dt = dt #if dt != "adaptive" else max(dts[:-1])
                    settings.adaptive = False #dt == "adaptive"
                    settings.sampling = strat(settings.spectrum)

                    states, exec_time, deficit = run(settings)
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

    #adaptive
    for k,strat in enumerate(sampling_strat):
        error_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        error_std_hm = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        deficit_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        mean_time_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        sanity_heatmap = [[0 for _ in range(len(n_sds))] for _ in range(len(dts))]
        for i, dt in enumerate(dts):
            for j, n_sd in enumerate(n_sds):
                sanity_heatmap[i][j] = "dt="+str(dt)+", n_sd="+str(n_sd)
                outputs = []
                deficits = []
                exec_times = []
                errors = []
                one_for_warmup = 1
                for it in range(iters_without_warmup + one_for_warmup):
                    settings = Settings(seed=it)

                    settings.n_sd = 2**n_sd
                    settings.dt = dt #if dt != "adaptive" else max(dts[:-1])
                    settings.adaptive = True
                    settings.sampling = strat(settings.spectrum)

                    states, exec_time, deficit = run(settings)
                    deficit *= settings.dt*settings.dv*settings.rho
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

                for key in outputs[0].keys():
                    mean_output[key] = sum((output[key] for output in outputs)) / len(
                        outputs
                    )
                mean_deficit = sum(deficits) / len(deficits)
                last_step_error = sum(errors) / len(errors)
                error_std = np.std(errors)

                plotter = SpectrumPlotter(settings, legend=False)
                plotter.fig = fig
                plotter.ax = axs[i, j]
                plotter.smooth = False
                # for step, vals in mean_output.items():
                #     error = plotter.plot(vals, step * settings.dt)
                # last_step_error = error/1e3 #grams to kg
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
        error_heatmaps_adaptive[sampling_strat_names[k]] = error_heatmap
        error_std_heatmaps_adaptive[sampling_strat_names[k]] = error_std_hm
        deficit_heatmaps_adaptive[sampling_strat_names[k]] = deficit_heatmap
        mean_time_heatmaps_adaptive[sampling_strat_names[k]] = mean_time_heatmap
        adaptive[sampling_strat_names[k]]["Error"] = error_heatmap
        adaptive[sampling_strat_names[k]]["Error_std"] = error_std_hm
        adaptive[sampling_strat_names[k]]["Deficit"] = deficit_heatmap
        adaptive[sampling_strat_names[k]]["MeanTime"] = mean_time_heatmap

    print(sanity_heatmap)
    if save is not None:
        n_sd = settings.n_sd
        plotter.save(save + "/" + f"{n_sd}_shima_fig_2" + "." + plotter.format)

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
    return regular,adaptive,sanity_heatmap


if __name__ == "__main__":
    regular,adaptive,type_matrix = main(plot=False, save=".")
    
results = {"regular":regular,"adaptive":adaptive,"type_matrix":type_matrix}
with open('test_runs_6_29.json', 'w', encoding='UTF-8') as f:
    json.dump(results, f)
