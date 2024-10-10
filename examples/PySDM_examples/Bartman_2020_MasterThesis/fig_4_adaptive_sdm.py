import os
import numpy as np

from matplotlib import pyplot as plt
from PySDM_examples.Shima_et_al_2009.example import run
from PySDM_examples.Shima_et_al_2009.settings import Settings
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumPlotter


def main(plot: bool = True, save: str = None):
    n_sds = [13, 15, 17, 19]
    dts = [10, 1, "adaptive"]
    iters_without_warmup = 5
    base_time = None
    base_error = None

    plt.ioff()
    fig, axs = plt.subplots(
        len(dts), len(n_sds), sharex=True, sharey=True, figsize=(10, 10)
    )

    for i, dt in enumerate(dts):
        for j, n_sd in enumerate(n_sds):
            outputs = []
            exec_times = []
            one_for_warmup = 1
            for it in range(iters_without_warmup + one_for_warmup):
                settings = Settings(seed=it)

                settings.n_sd = 2**n_sd
                settings.dt = dt if dt != "adaptive" else max(dts[:-1])
                settings.adaptive = dt == "adaptive"

                states, exec_time = run(settings)
                print(f"{dt=}, {n_sd=}, {exec_time=}, {it=}")
                exec_times.append(exec_time)
                outputs.append(states)
            mean_time = np.mean(exec_times[one_for_warmup:])
            if base_time is None:
                base_time = mean_time
            norm_time = mean_time / base_time
            mean_output = {}
            for key in outputs[0].keys():
                mean_output[key] = sum((output[key] for output in outputs)) / len(
                    outputs
                )

            plotter = SpectrumPlotter(settings, legend=False)
            plotter.fig = fig
            plotter.ax = axs[i, j]
            plotter.smooth = True
            for step, vals in mean_output.items():
                error = plotter.plot(vals, step * settings.dt)
            last_step_error = error
            if base_error is None:
                base_error = last_step_error
            norm_error = last_step_error / base_error

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
                f"time: {norm_time:.2f} error: {norm_error:.2f} (normalised)"
            )
            plotter.finished = False
            plotter.finish()
    if save is not None:
        n_sd = settings.n_sd
        plotter.save(save + "/" + f"{n_sd}_shima_fig_2" + "." + plotter.format)
    if plot:
        plotter.show()


if __name__ == "__main__":
    main(plot="CI" not in os.environ, save=".")
