import numpy as np
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot

from PySDM.physics import convert_to, si


def plot(
    var, qlabel, fname, output, vmin=None, vmax=None, cmin=None, cmax=None, line=None
):
    line = line or {15: ":", 20: "--", 25: "-", 30: "-."}
    dt = output["t"][1] - output["t"][0]
    dz = output["z"][1] - output["z"][0]
    tgrid = np.concatenate(((output["t"][0] - dt / 2,), output["t"] + dt / 2))
    zgrid = np.concatenate(((output["z"][0] - dz / 2,), output["z"] + dz / 2))
    convert_to(zgrid, si.km)

    fig = pyplot.figure(constrained_layout=True)
    gs = fig.add_gridspec(25, 5)
    ax1 = fig.add_subplot(gs[:-1, 0:4])
    if cmin is not None and cmax is not None:
        mesh = ax1.pcolormesh(
            tgrid, zgrid, output[var], cmap="BuPu", vmin=cmin, vmax=cmax
        )
    else:
        mesh = ax1.pcolormesh(tgrid, zgrid, output[var], cmap="BuPu")

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("z [km]")
    ax1.set_ylim(0, None)

    cbar = fig.colorbar(mesh, fraction=0.05, location="top")
    cbar.set_label(qlabel)

    ax2 = fig.add_subplot(gs[:-1, 4:], sharey=ax1)
    ax2.set_xlabel(qlabel)
    # ax2.set_yticks([], minor=False)

    last_t = 0
    for i, t in enumerate(output["t"]):
        x, z = output[var][:, i], output["z"].copy()
        convert_to(z, si.km)
        params = {"color": "black"}
        for line_t, line_s in line.items():
            if last_t < line_t * si.min <= t:
                params["ls"] = line_s
                ax2.plot(x, z, **params)
                if vmin is not None and vmax is not None:
                    ax1.axvline(t, ymin=vmin, ymax=vmax, **params)
                else:
                    ax1.axvline(t, **params)
        last_t = t

    show_plot(filename=fname, inline_format="png")


def plot_plusminus(
    plus_vars,
    minus_vars,
    qlabel,
    fname,
    output,
    vmin=None,
    vmax=None,
    line=None,
    cmin=0.0,
    cmax=None,
):
    line = line or {15: ":", 20: "--", 25: "-", 30: "-."}
    dt = output["t"][1] - output["t"][0]
    dz = output["z"][1] - output["z"][0]
    tgrid = np.concatenate(((output["t"][0] - dt / 2,), output["t"] + dt / 2))
    zgrid = np.concatenate(((output["z"][0] - dz / 2,), output["z"] + dz / 2))
    convert_to(zgrid, si.km)

    fig = pyplot.figure(constrained_layout=True)
    gs = fig.add_gridspec(25, 5)
    ax1 = fig.add_subplot(gs[:-1, 0:4])
    plot_var = 0.0 * output[plus_vars[0]]
    for plus_var in plus_vars:
        plot_var += output[plus_var]
    for minus_var in minus_vars:
        plot_var -= output[minus_var]

    mesh = ax1.pcolormesh(tgrid, zgrid, plot_var, cmap="BuPu", vmin=cmin, vmax=cmax)

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("z [km]")
    ax1.set_ylim(0, None)

    cbar = fig.colorbar(mesh, fraction=0.05, location="top")
    cbar.set_label(qlabel)

    ax2 = fig.add_subplot(gs[:-1, 4:], sharey=ax1)
    ax2.set_xlabel(qlabel)
    # ax2.set_yticks([], minor=False)

    last_t = 0
    for i, t in enumerate(output["t"]):
        x, z = plot_var[:, i], output["z"].copy()
        convert_to(z, si.km)
        params = {"color": "black"}
        for line_t, line_s in line.items():
            if last_t < line_t * si.min <= t:
                params["ls"] = line_s
                ax2.plot(x, z, **params)
                if vmin is not None and vmax is not None:
                    ax1.axvline(t, ymin=vmin, ymax=vmax, **params)
                else:
                    ax1.axvline(t, **params)
        last_t = t

    show_plot(filename=fname, inline_format="png")
