"""
Created at 20.08.2020
"""

from PySDM_examples.Shima_et_al_2009_Fig_2.settings import Settings
from PySDM_examples.Shima_et_al_2009_Fig_2.example import run
from PySDM_examples.Shima_et_al_2009_Fig_2.spectrum_plotter import SpectrumPlotter
from matplotlib import pyplot as plt


def main(plot: bool = True, save: str = None):
    n_sds = [13, 15, 17]
    dts = [10, 5, 1, 'adaptive']
    iters = 10
    base_time = None

    fig, axs = plt.subplots(len(dts), len(n_sds),
                            sharex=True, sharey=True, figsize=(10, 10))

    for i, dt in enumerate(dts):
        for j, n_sd in enumerate(n_sds):
            outputs = []
            exec_time = 0
            for iter in range(iters):
                settings = Settings()

                settings.n_sd = 2 ** n_sd
                settings.dt = dt if dt != 'adaptive' else 10
                settings.adaptive = dt == 'adaptive'

                states, exec_time = run(settings)
                outputs.append(states)
            mean_time = exec_time / iters
            if base_time is None:
                base_time = mean_time
            norm_time = mean_time / base_time
            mean_output = {}
            for key in outputs[0].keys():
                mean_output[key] = sum((output[key] for output in outputs)) / len(outputs)

            plotter = SpectrumPlotter(settings, legend=False)
            plotter.fig = fig
            plotter.ax = axs[i, j]
            plotter.smooth = True
            for step, vals in mean_output.items():
                plotter.plot(vals, step * settings.dt)

            plotter.ylabel = r'$\bf{dt: ' + str(dt) + '}$\ndm/dlnr [g/m^3/(unit dr/r)]' if j == 0 else None
            plotter.xlabel = 'particle radius [µm]\n' \
                             + r'$\bf{n_{sd}: 2^{' + str(n_sd) + '}}$' if i == len(dts) - 1 else None
            plotter.title = f'norm. time: {norm_time:.2f}; ' + plotter.title
            plotter.finished = False
            plotter.finish()
    if save is not None:
        n_sd = settings.n_sd
        plotter.save(save + "/" +
                     f"{n_sd}_shima_fig_2" +
                     "." + plotter.format)
    if plot:
        plotter.show()


if __name__ == '__main__':
    main(plot=True, save='.')