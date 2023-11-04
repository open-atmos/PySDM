from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import (
    SpectrumPlotter as SuperSpectrumPlotter,
)


class SpectrumPlotter(SuperSpectrumPlotter):
    def __init__(self, settings, title=None, grid=True, legend=False):
        size = 2 * 5.236
        pyplot.figure(num=1, figsize=(size, size * 0.54))
        pyplot.xlabel("particle radius [µm]")
        pyplot.ylabel("dm/dlnr [g/m^3/(unit dr/r)]")
        super().__init__(settings, title=title, grid=grid, legend=legend, log_base=2)
        self.color = None
        self.smooth = True

    @staticmethod
    def ticks():
        xticks = [4, 6.25, 12.5, 25, 50, 100, 200]
        pyplot.xticks(xticks, xticks)
        pyplot.yticks([0.5 * i for i in range(5)], [0, None, 1, None, 2])

    def show(self):
        self.finish()
        self.ticks()
        show_plot()

    def plot(self, spectrum, t):
        settings = self.settings
        self.plot_data(settings, t, spectrum)
