"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import VBox, Box, Play, Output, IntSlider, jslink, Layout
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PySDM_examples.ICMW_2012_case_1 import plotter
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
        self.plots_box = Box(
            children=tuple(self.plots.values()),
            layout=Layout(display='flex', flex_flow='column')
        )

        self.reinit({})

    def clear(self):
        self.plots_box.children = ()

    def reinit(self, products):
        self.products = products

        self.plots.clear()
        for var in products.keys():
            self.plots[var] = Output()
        self.ims = {}

        self.nans = np.full((self.setup.grid[0], self.setup.grid[1]), np.nan)  # TODO: np.nan
        for key in self.plots.keys():
            with self.plots[key]:
                clear_output()
                _, ax = plt.subplots(1, 1)
                product = self.products[key]
                self.ims[key] = plotter.image(ax, self.nans, self.setup.size,
                                              label=f"{product.description} [{product.unit}]",
                                              # cmap=self.clims[key][2], # TODO: Reds, Blues, YlGnBu...
                                              scale=product.scale
                                              )
                self.ims[key].set_clim(vmin=product.range[0], vmax=product.range[1])
                plt.show()

        self.plots_box.children = tuple(self.plots.values())
        n_steps = len(self.setup.steps)
        self.step_slider.max = n_steps - 1
        self.play.max = n_steps - 1
        self.play.value = 0
        self.step_slider.value = 0
        self.replot(step=0)

    def handle_replot(self, bunch):
        self.replot(bunch.new)

    def replot(self, step):
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
        self.play.observe(self.handle_replot, 'value')
        return VBox([
            Box([self.play, self.step_slider, self.fps_slider]),
            self.plots_box
        ])
