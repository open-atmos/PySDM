"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import VBox, HBox, Box, Play, Output, IntSlider, jslink
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import numpy as np


class DemoViewer:
    def __init__(self, storage, setup):
        self.storage = storage
        self.setup = setup

        self.nans = None

        # TODO: fps slider
        self.play = Play()
        self.slider = IntSlider()
        self.plots = {
            "m0": Output(),
            "th": Output(),
            "qv": Output()
        }
        self.ims = {}

        self.reinit()

    def reinit(self):
        n_steps = len(self.setup.steps)
        self.slider.max = n_steps - 1
        self.play.max = n_steps - 1
        self.play.value = 0
        self.slider.value = 0
        self.nans = np.full((self.setup.grid[0], self.setup.grid[1]), np.nan)
        for key in self.plots.keys():
            with self.plots[key]:
                clear_output()
                self.ims[key] = plt.imshow(self.nans, cmap='GnBu')
                self.ims[key].set_clim(vmin=0, vmax=500000000)
                plt.colorbar()
                plt.title(key)
                plt.show()

    def replot(self, bunch):
        step = bunch.new

        for key in self.plots.keys():
            try:
                data = self.storage.load(self.setup.steps[step], key)
            except self.storage.Exception:
                data = self.nans
            self.ims[key].set_data(data)

        for key in self.plots.keys():
            with self.plots[key]:
                clear_output(wait=True)
                display(self.ims[key].figure)

    def box(self):
        jslink((self.play, 'value'), (self.slider, 'value'))
        self.play.observe(self.replot, 'value')
        return VBox([Box([self.play, self.slider]), HBox(tuple(self.plots.values()))])
