"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import VBox, HBox, Box, Play, Output, IntSlider, jslink
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from examples.ICMW_2012_case_1 import plotter
import numpy as np


class DemoViewer:
    def __init__(self, storage, setup):
        self.storage = storage
        self.setup = setup

        self.nans = None

        self.play = Play()
        self.step_slider = IntSlider()
        self.fps_slider = IntSlider(min=100, max=1000, description="1000/fps")
        self.plots = {}
        for var in setup.output_vars:
            self.plots[var] = Output() 
        self.ims = {}

        self.reinit()


    def reinit(self):
        n_steps = len(self.setup.steps)
        self.step_slider.max = n_steps - 1
        self.play.max = n_steps - 1
        self.play.value = 0
        self.step_slider.value = 0
        self.clims = { # TODO !
            "m0": (0, 1e8,  'YlGnBu'),
            "th": (295,305, 'Reds'),
            "qv": (0,1e-2,  'Greens'),
            "RH": (0,1.2,   'GnBu'),
            "x_m1": (1e-8, 1e-7, 'Reds')
        }

        self.nans = np.full((self.setup.grid[0], self.setup.grid[1]), np.nan) # TODO: np.nan
        for key in self.plots.keys():
            with self.plots[key]:
                clear_output()
                _, ax = plt.subplots(1, 1)
                self.ims[key] = plotter.image(ax, self.nans, self.setup.size, label=key, cmap=self.clims[key][2])
                self.ims[key].set_clim(vmin = self.clims[key][0], vmax = self.clims[key][1])
                #plt.colorbar()
                #plt.title(key)
                plt.show()

    def replot(self, bunch):
        step = bunch.new

        for key in self.plots.keys():
            try:
                data = self.storage.load(self.setup.steps[step], key)
            except self.storage.Exception:
                data = self.nans
            plotter.image_update(self.ims[key], data)

        for key in self.plots.keys():
            with self.plots[key]:
                clear_output(wait=True)
                display(self.ims[key].figure)

    def box(self):
        jslink((self.play, 'value'), (self.step_slider, 'value'))
        jslink((self.play, 'interval'), (self.fps_slider, 'value'))
        self.play.observe(self.replot, 'value')
        return VBox([Box([self.play, self.step_slider, self.fps_slider]), HBox(tuple(self.plots.values()))])
