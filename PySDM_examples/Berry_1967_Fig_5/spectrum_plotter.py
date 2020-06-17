"""
Created at 12.08.2019
"""

import numpy as np
from matplotlib import pyplot
from PySDM_examples.Shima_et_al_2009_Fig_2.spectrum_plotter import SpectrumPlotter as SuperSpectrumPlotter
from labellines import labelLines


class SpectrumPlotter(SuperSpectrumPlotter):
    def __init__(self, setup):
        super().__init__(setup)
        self.color = 'grey'
        self.smooth = True
        size = 2*5.236
        pyplot.figure(num=1, figsize=(size, size * 0.54))

    def show(self, title=None, legend=False, grid=True):
        pyplot.xscale('log', basex=2)
        xticks = [4, 6.25, 12.5, 25, 50, 100, 200]
        pyplot.xticks(xticks, xticks)
        pyplot.yticks([0.5 * i for i in range(5)], [0, None, 1, None, 2])
        pyplot.xlabel('particle radius [µm]')
        pyplot.ylabel('dm/dlnr [g/m^3/(unit dr/r)]')
        if legend:
            labelLines(pyplot.gca().get_lines(), zorder=2.5)
        if grid:
            pyplot.grid()
        pyplot.title(title)
        pyplot.show()

    def plot(self, spectrum, t):
        setup = self.setup
        self.plot_data(setup, t, spectrum)


