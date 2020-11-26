"""
Created at 2019
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class _Plot:

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.ax.set_title(' ')


class _ImagePlot(_Plot):
    line_args = {'color': 'red', 'alpha': .75, 'linestyle': ':', 'linewidth': 5}

    def __init__(self, fig, ax, grid, size, product, show=False, lines=False, cmap='YlGnBu'):
        super().__init__(fig, ax)
        self.nans = np.full(grid, np.nan)

        self.dx = size[0] / grid[0]
        self.dz = size[1] / grid[1]

        xlim = (0, size[0])
        zlim = (0, size[1])

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(zlim)

        if lines:
            self.lines = {'X': [None]*2, 'Z': [None]*2}
            self.lines['X'][0] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
            self.lines['Z'][0] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]
            self.lines['X'][1] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
            self.lines['Z'][1] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]

        data = np.full_like(self.nans, product.range[0])
        label = f"{product.description} [{product.unit}]"
        scale = product.scale

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')

        self.im = self.ax.imshow(self._transpose(data),
                                 origin='lower',
                                 extent=(*xlim, *zlim),
                                 cmap=cmap,
                                 norm=matplotlib.colors.LogNorm() if scale == 'log' and np.isfinite(
                                     data).all() else None # TODO: this is always None!!!
                                 )
        plt.colorbar(self.im, ax=self.ax).set_label(label)
        self.im.set_clim(vmin=product.range[0], vmax=product.range[1])
        if show:
            plt.show()

    @staticmethod
    def _transpose(data):
        if data is not None:
            return data.T

    def update(self, data,  step):
        data = self._transpose(data)
        if data is not None:
            self.im.set_data(data)
            self.ax.set_title(f"min:{np.amin(data): .3g}    max:{np.amax(data): .3g}    t/dt:{step: >6}")

    def update_lines(self, focus_x, focus_z):
        self.lines['X'][0].set_xdata(x=focus_x[0] * self.dx)
        self.lines['Z'][0].set_ydata(y=focus_z[0] * self.dz)
        self.lines['X'][1].set_xdata(x=focus_x[1] * self.dx)
        self.lines['Z'][1].set_ydata(y=focus_z[1] * self.dz)


class _SpectrumPlot(_Plot):

    def __init__(self, r_bins, show=True):
        super().__init__(*plt.subplots(1, 1))
        self.ax.set_xlim(np.amin(r_bins), np.amax(r_bins))
        self.ax.set_xlabel("particle radius [μm]")
        self.ax.set_ylabel("specific concentration density [mg$^{-1}$ μm$^{-1}$]")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_ylim(1, 5e3)
        self.ax.grid(True)
        self.spec_wet = self.ax.step(r_bins, np.full_like(r_bins, np.nan), label='wet')[0]
        self.spec_dry = self.ax.step(r_bins, np.full_like(r_bins, np.nan), label='dry')[0]
        self.ax.legend()
        if show:
            plt.show()

    def update_wet(self, data, step):
        self.spec_wet.set_ydata(data)
        self.ax.set_title(f"t/dt:{step}")

    def update_dry(self, dry):
        self.spec_dry.set_ydata(dry)


class _TimeseriesPlot(_Plot):

    def __init__(self, fig, ax, times, show=True):
        super().__init__(fig, ax)
        self.ax.set_xlim(0, times[-1])
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("rainfall [mm/day]")
        self.ax.set_ylim(0, 1e-1)
        self.ax.grid(True)
        self.ydata = np.full_like(times, np.nan, dtype=float)
        self.timeseries = self.ax.step(times, self.ydata, where='pre')[0]
        if show:
            plt.show()

    def update(self, data):
        self.ydata[0:len(data)] = data[:]
        self.timeseries.set_ydata(self.ydata)