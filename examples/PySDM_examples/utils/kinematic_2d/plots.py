import matplotlib.pyplot as plt
import numpy as np

from PySDM.physics import constants as const


class _Plot:
    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.ax.set_title(" ")


class _ImagePlot(_Plot):
    line_args = {"color": "red", "alpha": 0.666, "linestyle": ":", "linewidth": 3}

    def __init__(
        self, fig, ax, grid, size, product, show=False, lines=False, cmap="YlGnBu"
    ):
        super().__init__(fig, ax)
        self.nans = np.full(grid, np.nan)

        self.dx = size[0] / grid[0]
        self.dz = size[1] / grid[1]

        xlim = (0, size[0])
        zlim = (0, size[1])

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(zlim)

        if lines:
            self.lines = {"X": [None] * 2, "Z": [None] * 2}
            self.lines["X"][0] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
            self.lines["Z"][0] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]
            self.lines["X"][1] = plt.plot([-1] * 2, zlim, **self.line_args)[0]
            self.lines["Z"][1] = plt.plot(xlim, [-1] * 2, **self.line_args)[0]

        data = np.full_like(self.nans, np.nan)
        label = f"{product.name} [{product.unit}]"

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Z [m]")

        self.im = self.ax.imshow(
            self._transpose(data), origin="lower", extent=(*xlim, *zlim), cmap=cmap
        )
        plt.colorbar(self.im, ax=self.ax).set_label(label)
        if show:
            plt.show()

    @staticmethod
    def _transpose(data):
        if data is not None:
            return data.T
        return None

    def update(self, data, step, data_range):
        data = self._transpose(data)
        if data is not None:
            self.im.set_data(data)
            if data_range is not None:
                self.im.set_clim(vmin=data_range[0], vmax=data_range[1])
            nanmin = np.nan
            nanmax = np.nan
            if np.isfinite(data).any():
                nanmin = np.nanmin(data)
                nanmax = np.nanmax(data)
            self.ax.set_title(
                f"min:{nanmin: .3g}    max:{nanmax: .3g}    t/dt_out:{step: >6}"
            )

    def update_lines(self, focus_x, focus_z):
        self.lines["X"][0].set_xdata(x=(focus_x[0] + 0.15) * self.dx)
        self.lines["Z"][0].set_ydata(y=(focus_z[0] + 0.15) * self.dz)
        self.lines["X"][1].set_xdata(x=(focus_x[1] - 0.25) * self.dx)
        self.lines["Z"][1].set_ydata(y=(focus_z[1] - 0.25) * self.dz)


class _SpectrumPlot(_Plot):
    def __init__(self, r_bins, initial_spectrum_per_mass_of_dry_air, show=True):
        super().__init__(*plt.subplots(1, 1))
        self.ax.set_xlim(np.amin(r_bins), np.amax(r_bins))
        self.ax.set_xlabel("particle radius [μm]")
        self.ax.set_ylabel("specific concentration density [mg$^{-1}$ μm$^{-1}$]")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_ylim(1, 5e3)
        self.ax.grid(True)
        vals = initial_spectrum_per_mass_of_dry_air.size_distribution(
            r_bins * const.si.um
        )
        const.convert_to(vals, const.si.mg**-1 / const.si.um)
        self.ax.plot(r_bins, vals, label="spectrum sampled at t=0")
        self.spec_wet = self.ax.step(
            r_bins,
            np.full_like(r_bins, np.nan),
            label="binned super-particle wet sizes",
        )[0]
        self.spec_dry = self.ax.step(
            r_bins,
            np.full_like(r_bins, np.nan),
            label="binned super-particle dry sizes",
        )[0]
        self.ax.legend()
        if show:
            plt.show()

    def update_wet(self, data, step):
        self.spec_wet.set_ydata(data)
        self.ax.set_title(f"t/dt_out:{step}")

    def update_dry(self, dry):
        self.spec_dry.set_ydata(dry)


class _TimeseriesPlot(_Plot):
    def __init__(self, fig, ax, times, show=True):
        super().__init__(fig, ax)
        self.ax.set_xlim(0, times[-1])
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("rainfall [mm/day]")
        self.ax.grid(True)
        self.ydata = np.full_like(times, np.nan, dtype=float)
        self.timeseries = self.ax.step(times, self.ydata, where="pre")[0]
        if show:
            plt.show()

    def update(self, data, data_range):
        if data is not None:
            self.ydata[0 : len(data)] = data[:]
            if data_range[0] != data_range[1]:
                self.ax.set_ylim(data_range[0], 1.1 * data_range[1])
        else:
            self.ydata[:] = np.nan
        self.timeseries.set_ydata(self.ydata)


class _TemperaturePlot(_Plot):
    def __init__(self, T_bins, formulae, show=True):
        super().__init__(*plt.subplots(1, 1))
        self.formulae = formulae
        self.ax.set_xlim(np.amax(T_bins), np.amin(T_bins))
        self.ax.set_xlabel("temperature [K]")
        self.ax.set_ylabel("freezable fraction / cdf [1]")
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.grid(True)
        # self.ax.plot(T_bins, self.formulae.freezing_temperature_spectrum.cdf(T_bins),
        #              label=str(self.formulae.freezing_temperature_spectrum) + " (sampled at t=0)")
        self.spec = self.ax.step(
            T_bins,
            np.full_like(T_bins, np.nan),
            label="binned super-particle attributes",
            where="mid",
        )[0]
        self.ax.legend()
        if show:
            plt.show()

    def update(self, data, step):
        self.ax.set_title(f"t/dt_out:{step}")
        self.spec.set_ydata(data)


class _TerminalVelocityPlot(_Plot):
    def __init__(self, radius_bins, formulae, show=True):
        self.formulae = formulae
        super().__init__(*plt.subplots(1, 1))

        self.ax.set_xlim(
            np.amin(radius_bins) / const.si.um, np.amax(radius_bins) / const.si.um
        )
        self.ax.set_xlabel("radius [μm]")
        self.ax.set_ylabel("mean terminal velocity [m/s]")
        self.ax.set_ylim(0, 0.1)
        self.ax.grid(True)

        self.radius_bins = radius_bins
        # self.ax.plot(T_bins, self.formulae.freezing_temperature_spectrum.cdf(T_bins),
        #              label=str(self.formulae.freezing_temperature_spectrum) + " (sampled at t=0)")
        # nans = np.full_like(radius_bins[:-1], np.nan)
        # self.spec = self.ax.fill_between(
        #     (radius_bins[:-1] + np.diff(radius_bins)/2) / const.si.um,
        #     nans,
        #     nans,
        #     marker='o'
        # )[0]
        # label='binned super-particle attributes',
        # where='mid'
        # )[0]
        # self.ax.legend()

        if show:
            plt.show()

    def update(self, data_min, data_max, step):
        self.ax.set_title(f"t/dt_out:{step}")
        self.ax.collections.clear()
        self.ax.fill_between(
            (self.radius_bins[:-1] + np.diff(self.radius_bins) / 2) / const.si.um,
            data_min,
            data_max,
            color="gray",
        )
        # self.spec.set_ydata(data)
