"""
Created at 2019
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class _Plot:

    def __init__(self, *, fig_kw={}):
        self.fig, self.ax = plt.subplots(1, 1, **fig_kw)
        self.ax.set_title(' ')


class _ImagePlot(_Plot):
    line_args = {'color': 'red', 'alpha': .75, 'linestyle': ':', 'linewidth': 5}

    def __init__(self, grid, size, product):
        super().__init__()
        self.nans = np.full(grid, np.nan)

        self.dx = size[0] / grid[0]
        self.dz = size[1] / grid[1]

        xlim = (0, size[0])
        zlim = (0, size[1])

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(zlim)

        self.lines = {'X': [None]*2, 'Z': [None]*2}
        self.lines['X'][0] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
        self.lines['Z'][0] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]
        self.lines['X'][1] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
        self.lines['Z'][1] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]

        data = self.nans
        cmap = 'YlGnBu'
        label = f"{product.description} [{product.unit}]"
        scale = product.scale

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')

        self.im = self.ax.imshow(self._transpose(data),
                                 origin='lower',
                                 extent=(*xlim, *zlim),
                                 cmap=cmap,
                                 norm=matplotlib.colors.LogNorm() if scale == 'log' and np.isfinite(
                                     data).all() else None
                                 )
        plt.colorbar(self.im, ax=self.ax).set_label(label)
        self.im.set_clim(vmin=product.range[0], vmax=product.range[1])

        plt.show()

    @staticmethod
    def _transpose(data):
        if data is not None:
            return data.T

    def update(self, data, focus_x, focus_z, step):
        data = self._transpose(data)
        if data is not None:
            self.im.set_data(data)
            self.ax.set_title(f"min:{np.amin(data):.4g}    max:{np.amax(data):.4g}    t/dt:{step}")

        self.lines['X'][0].set_xdata(x=focus_x[0] * self.dx)
        self.lines['Z'][0].set_ydata(y=focus_z[0] * self.dz)
        self.lines['X'][1].set_xdata(x=focus_x[1] * self.dx)
        self.lines['Z'][1].set_ydata(y=focus_z[1] * self.dz)


class _SpectrumPlot(_Plot):

    def __init__(self, r_bins):
        super().__init__()
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
        plt.show()

    def update_wet(self, data, step):
        self.spec_wet.set_ydata(data)
        self.ax.set_title(f"t/dt:{step}")

    def update_dry(self, dry):
        self.spec_dry.set_ydata(dry)


class _TimeseriesPlot(_Plot):

    def __init__(self, steps, dt):
        default_figsize = matplotlib.rcParams["figure.figsize"]
        super().__init__(fig_kw={'figsize': (2.25*default_figsize[0], default_figsize[1]/2)})
        self.ax.set_xlim(0, dt * steps[-1])
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("rainfall [mm/day]")
        self.ax.set_ylim(0, 1e-1)
        self.ax.grid(True)
        self.ydata = np.full_like(steps, np.nan, dtype=float)
        self.timeseries = self.ax.step(steps, self.ydata, where='pre')[0]
        plt.show()

    def update(self, data):
        self.ydata[0:len(data)] = data[:]
        self.timeseries.set_ydata(self.ydata)