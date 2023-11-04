import numpy as np
from matplotlib import colors, pyplot

from PySDM.physics.constants import convert_to, si


def log_kwargs(clog, cmin, cmax):
    if clog:
        return {"norm": colors.LogNorm(vmin=cmin, vmax=cmax)}
    return {"vmin": cmin, "vmax": cmax}


def plot_ax(
    ax,
    var,
    qlabel,
    output,
    contour_var1=None,
    contour_lvl1=None,
    contour_var2=None,
    contour_lvl2=None,
    cmin=None,
    cmax=None,
    clog=False,
):
    tgrid = output["t"].copy()
    zgrid = output["z"].copy()
    convert_to(zgrid, si.km)

    if clog:
        data = output[var].copy()
        data[data == 0] = np.nan
    else:
        data = output[var]

    mesh = ax.pcolormesh(
        tgrid,
        zgrid,
        data,
        cmap="BuPu",
        shading="nearest",
        **log_kwargs(clog, cmin, cmax),
    )

    if contour_var1 is not None and contour_lvl1 is not None:
        ax.contour(
            tgrid,
            zgrid,
            output[contour_var1],
            contour_lvl1,
            colors="k",
            linestyles="--",
        )
    if contour_var2 is not None and contour_lvl2 is not None:
        ax.contour(
            tgrid,
            zgrid,
            output[contour_var2],
            contour_lvl2,
            colors="r",
            linestyles="--",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [km]")
    ax.set_ylim(0, None)

    if clog:
        cbar_levels = np.logspace(np.log10(cmin), np.log10(cmax), 5, endpoint="True")
    else:
        cbar_levels = np.linspace(cmin, cmax, 5, endpoint="True")
    cbar = pyplot.colorbar(mesh, fraction=0.05, location="top", extend="max", ax=ax)
    cbar.set_ticks(cbar_levels)
    cbar.set_label(qlabel)


def plot_zeros_ax(ax, var, qlabel, output, cmin=None, cmax=None, clog=False):
    dt = output["t"][1] - output["t"][0]
    dz = output["z"][1] - output["z"][0]
    tgrid = np.concatenate(((output["t"][0] - dt / 2,), output["t"] + dt / 2))
    zgrid = np.concatenate(((output["z"][0] - dz / 2,), output["z"] + dz / 2))
    convert_to(zgrid, si.km)

    zeros = np.zeros_like(output[var])
    kwargs = log_kwargs(clog, cmin, cmax)
    cmap = "BuPu"
    mesh = ax.pcolormesh(tgrid, zgrid, zeros, cmap=cmap, **kwargs)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [km]")
    ax.set_ylim(0, None)

    cbar = pyplot.colorbar(mesh, fraction=0.05, location="top", ax=ax)
    cbar.set_label(qlabel)
