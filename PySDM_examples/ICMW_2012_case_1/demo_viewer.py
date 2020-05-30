"""
Created at 02.10.2019
"""

from ipywidgets import VBox, Box, Play, Output, IntSlider, IntRangeSlider, jslink, Layout, HBox, Select
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import numpy as np


def _transform(data):
    return data.T


class DemoViewer:
    def __init__(self, storage, setup):
        self.storage = storage
        self.setup = setup

        self.nans = None

        self.play = Play()
        self.step_slider = IntSlider()
        self.fps_slider = IntSlider(min=100, max=1000, description="1000/fps")
        self.product_select = Select()
        self.plots_box = Box()

        self.slider = {}
        self.lines = {'x': [{}, {}], 'y': [{}, {}]}
        for xy in ('x', 'y'):
            self.slider[xy] = IntRangeSlider(min=0, max=1, description=f'spectrum_{xy}',
                                             orientation='horizontal' if xy == 'x' else 'vertical')

        self.reinit({})

    def clear(self):
        self.plots_box.children = ()

    def reinit(self, products):
        self.products = products
        self.product_select.options = [key for key, val in products.items() if len(val.shape) == 2]
        self.plots = {}
        for var in products.keys():
            self.plots[var] = Output()
        self.ims = {}
        self.axs = {}
        self.figs = {}
        for j, xy in enumerate(('x', 'y')):
            self.slider[xy].max = self.setup.grid[j]

        self.nans = np.full((self.setup.grid[0], self.setup.grid[1]), np.nan)  # TODO: np.nan

        for key in self.plots.keys():
            with self.plots[key]:
                clear_output()
                product = self.products[key]
                if len(product.shape) == 2:

                    data=self.nans
                    domain_size_in_metres=self.setup.size
                    cmap='YlGnBu'
                    fig, ax = plt.subplots(1, 1)
                    label = f"{product.description} [{product.unit}]"
                    scale = product.scale

                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Z [m]')
                    im = ax.imshow(_transform(data),
                                   origin='lower',
                                   extent=(0, domain_size_in_metres[0], 0, domain_size_in_metres[1]),
                                   cmap=cmap,
                                   norm=matplotlib.colors.LogNorm() if scale == 'log' and np.isfinite(
                                       data).all() else None
                                   )
                    plt.colorbar(im, ax=ax).set_label(label)
                    im.set_clim(vmin=product.range[0], vmax=product.range[1])

                    x = self.slider['x'].value[0] * self.setup.size[0] / self.setup.grid[0]
                    y = self.slider['y'].value[0] * self.setup.size[1] / self.setup.grid[1]
                    self.lines['x'][0][key] = ax.axvline(x=x, color='red')
                    self.lines['y'][0][key] = ax.axhline(y=y, color='red')
                    x = self.slider['x'].value[1] * self.setup.size[0]/self.setup.grid[0]
                    y = self.slider['y'].value[1] * self.setup.size[1]/self.setup.grid[1]
                    self.lines['x'][1][key] = ax.axvline(x=x, color='red')
                    self.lines['y'][1][key] = ax.axhline(y=y, color='red')
                elif len(product.shape) == 3:
                    fig, ax = plt.subplots(1, 1)
                    ax.set_xlim(np.amin(self.setup.v_bins), np.amax(self.setup.v_bins))
                    ax.set_ylim(0, 10)
                    ax.set_xlabel("TODO [TODO]")
                    ax.set_ylabel("TODO [TODO]")
                    ax.set_xscale('log')
                    ax.grid(True)
                    im = ax.step(self.setup.v_bins[:-1], np.full_like(self.setup.v_bins[:-1], np.nan))
                    im = im[0]
                else:
                    raise NotImplementedError()
                self.figs[key], self.ims[key], self.axs[key] = fig, im, ax
                plt.show()

        self.plot_box = Box()
        if len(products.keys()) > 0:
            self.plots_box.children = (
                HBox(children=(self.slider['y'], VBox((self.slider['x'], self.plot_box)))),
                self.plots['Particles Size Spectrum']
            )

        n_steps = len(self.setup.steps)
        self.step_slider.max = n_steps - 1
        self.play.max = n_steps - 1
        self.play.value = 0
        self.step_slider.value = 0
        self.replot()

    def replot(self, _=None):
        if self.product_select.value in self.plots:
            self.plot_box.children = [self.plots[self.product_select.value]]

        step = self.step_slider.value
        for key in self.plots.keys():
            try:
                data = self.storage.load(self.setup.steps[step], key)
            except self.storage.Exception:
                data = self.nans
            if len(self.products[key].shape) == 2:
                self.ims[key].set_data(_transform(data))
                self.axs[key].set_title(f"min:{np.amin(data):.4g}    max:{np.amax(data):.4g}    std:{np.std(data):.4g}")

                self.lines['x'][0][key].set_xdata(x=self.slider['x'].value[0] * self.setup.size[0]/self.setup.grid[0])
                self.lines['y'][0][key].set_ydata(y=self.slider['y'].value[0] * self.setup.size[1]/self.setup.grid[1])
                self.lines['x'][1][key].set_xdata(x=self.slider['x'].value[1] * self.setup.size[0]/self.setup.grid[0])
                self.lines['y'][1][key].set_ydata(y=self.slider['y'].value[1] * self.setup.size[1]/self.setup.grid[1])
            elif len(self.products[key].shape) == 3:
                xrange = slice(*self.slider['x'].value)
                yrange = slice(*self.slider['y'].value)
                data = data[xrange, yrange, :]
                data = np.mean(np.mean(data, axis=0), axis=0)
                self.ims[key].set_ydata(data)
                amax = np.amax(data)
                if np.isfinite(amax):
                    self.axs[key].set_ylim((0, amax))
            else:
                raise NotImplementedError()

        for key in self.plots.keys():
            with self.plots[key]:
                clear_output(wait=True)
                display(self.figs[key])

    def box(self):
        jslink((self.play, 'value'), (self.step_slider, 'value'))
        jslink((self.play, 'interval'), (self.fps_slider, 'value'))
        self.play.observe(self.replot, 'value')
        self.product_select.observe(self.replot, 'value')
        for xy in ('x', 'y'):
            self.slider[xy].observe(self.replot, 'value')
        return VBox([
            Box([self.play, self.step_slider, self.fps_slider]),
            self.product_select,
            self.plots_box
        ])
