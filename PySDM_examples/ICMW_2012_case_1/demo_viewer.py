"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import VBox, Box, Play, Output, IntSlider, jslink, Layout, HBox
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

        self.xslider1 = IntSlider(min=0, max=0, description='spectrum_x1')
        self.yslider1 = IntSlider(min=0, max=0, description='spectrum_y1')
        self.xslider2 = IntSlider(min=0, max=0, description='spectrum_x2')
        self.yslider2 = IntSlider(min=0, max=0, description='spectrum_y2')
        self.coordinate_select1 = VBox([self.xslider1, self.yslider1])
        self.coordinate_select2 = VBox([self.xslider2, self.yslider2])


        self.reinit({})


    def clear(self):
        self.plots_box.children = ()

    def reinit(self, products):
        self.products = products

        self.plots.clear()
        for var in products.keys():
            if var == 'Particles Size Spectrum': # TODO check dimensionality of product to be 2d instead
                continue
            self.plots[var] = Output()
        self.ims = {}
        self.axs = {}
        self.spectrum = Output()
        self.xslider1.max = self.setup.grid[0] - 1
        self.yslider1.max = self.setup.grid[1] - 1
        self.xslider2.max = self.setup.grid[0] - 1
        self.yslider2.max = self.setup.grid[1] - 1

        self.nans = np.full((self.setup.grid[0], self.setup.grid[1]), np.nan)# TODO: np.nan
        with self.spectrum:
            self.spectrum_figure, ax = plt.subplots(1,1)
            self.spectrum_plot =  ax.step(self.setup.v_bins[:-1], np.full_like(self.setup.v_bins[:-1], np.nan), where='post')[0]
            plt.show()
        self.vlines1, self.hlines1, self.vlines2, self.hlines2 = {},{},{},{}
        for key in self.plots.keys():
            with self.plots[key]:
                clear_output()
                fig, ax = plt.subplots(1, 1)
                product = self.products[key]
                self.ims[key], self.axs[key] = plotter.image(ax, self.nans, self.setup.size,
                                              label=f"{product.description} [{product.unit}]",
                                              # cmap=self.clims[key][2], # TODO: Reds, Blues, YlGnBu...
                                              scale=product.scale
                                              )
                self.ims[key].set_clim(vmin=product.range[0], vmax=product.range[1])
                self.vlines1[key] = self.axs[key].axvline(x=self.xslider1.value * self.setup.size[0]/self.setup.grid[0], color='red')
                self.hlines1[key] = self.axs[key].axhline(y=self.yslider1.value * self.setup.size[1]/self.setup.grid[1], color='red')

                self.vlines2[key] = self.axs[key].axvline(x=self.xslider2.value * self.setup.size[0]/self.setup.grid[0], color='red')
                self.hlines2[key] = self.axs[key].axhline(y=self.yslider2.value * self.setup.size[1]/self.setup.grid[1], color='red')
                plt.show()

        self.plots_box.children = [self.coordinate_select1, self.coordinate_select2, *self.plots.values()]
        n_steps = len(self.setup.steps)
        self.step_slider.max = n_steps - 1
        self.play.max = n_steps - 1
        self.play.value = 0
        self.step_slider.value = 0
        self.replot()



    def replot(self, _ = None):
        step = self.step_slider.value
        for key in self.plots.keys():
            try:
                data = self.storage.load(self.setup.steps[step], key)
            except self.storage.Exception:
                data = self.nans
            plotter.image_update(self.ims[key], self.axs[key], data)

            self.vlines1[key].set_xdata(x=self.xslider1.value * self.setup.size[0]/self.setup.grid[0])
            self.hlines1[key].set_ydata(y=self.yslider1.value * self.setup.size[1]/self.setup.grid[1])
            self.vlines2[key].set_xdata(x=self.xslider2.value * self.setup.size[0]/self.setup.grid[0])
            self.hlines2[key].set_ydata(y=self.yslider2.value * self.setup.size[1]/self.setup.grid[1])

        self.plot_spectra()
        for key in self.plots.keys():
            with self.plots[key]:
                clear_output(wait=True)
                display(self.ims[key].figure)

    def plot_spectra(self):
        step = self.step_slider.value
        xrange = slice(self.xslider1.value, self.xslider2.value)
        yrange = slice(self.yslider1.value, self.yslider2.value)
        vbins = self.setup.v_bins

        try:
            data = self.storage.load(self.setup.steps[step], 'Particles Size Spectrum')[xrange, yrange,:]
            data = np.mean(np.mean(data, axis=0), axis=0)
        except self.storage.Exception:
            data = np.full_like(vbins[:-1], np.nan)
        plt.step(vbins[:-1], data)
        plt.xscale('log')
        plt.xlabel('r')
        plt.ylabel('n')
        # self.spectrum_plot.set_xdata(x = vbins[:-1])
        # self.spectrum_plot.set_ydata(y = data)
        with self.spectrum:
            clear_output(wait=True)
            display(self.spectrum_figure)





    def box(self):
        jslink((self.play, 'value'), (self.step_slider, 'value'))
        jslink((self.play, 'interval'), (self.fps_slider, 'value'))
        self.play.observe(self.replot, 'value')
        self.xslider1.observe(self.replot, 'value')
        self.yslider1.observe(self.replot, 'value')
        self.xslider2.observe(self.replot, 'value')
        self.yslider2.observe(self.replot, 'value')
        return VBox([
            Box([self.play, self.step_slider, self.fps_slider]),
            self.plots_box, self.spectrum
        ])



