import matplotlib
import numpy as np
from matplotlib import pyplot
from PySDM_examples.Arabas_et_al_2023.curved_text import CurvedText
from PySDM_examples.Arabas_et_al_2023.frozen_fraction import FrozenFraction

from PySDM.physics import si

labels = {True: "singular/INAS", False: "time-dependent/ABIFM"}
colors = {True: "black", False: "teal"}
qi_unit = si.g / si.m**3


def make_temperature_plot(data):
    pyplot.xlabel("time [s]")

    xy1 = pyplot.gca()

    xy1.set_ylabel("temperature [K]")
    xy1.set_ylim(200, 300)
    datum = data[0]["products"]
    xy1.plot(datum["t"], datum["T"], marker=".", label="T", color="black")

    xy2 = xy1.twinx()
    plotted = {singular: False for singular in (True, False)}
    for v in data:
        datum = v["products"]
        xy2.plot(
            datum["t"],
            np.asarray(datum["qi"]) / qi_unit,  # marker='.',
            label=(
                f"Monte-Carlo ({labels[v['singular']]})"
                if not plotted[v["singular"]]
                else ""
            ),
            color=colors[v["singular"]],
        )
        plotted[v["singular"]] = True
    xy2.set_ylabel("ice water content [g/m3]")

    xy1.grid()
    xy1.legend()  # bbox_to_anchor=(.2, 1.15))
    xy2.legend()  # bbox_to_anchor=(.7, 1.15))


def make_freezing_spec_plot(
    data,
    formulae,
    volume,
    droplet_volume,
    total_particle_number,
    surf_spec,
    cooling_rate_K_min=None,
):
    pyplot.xlabel("temperature [K]")
    plotted = {singular: False for singular in (True, False)}

    prim = pyplot.gca()
    for v in data:
        datum = v["products"]
        color = colors[v["singular"]]
        prim.plot(
            datum["T"],
            np.asarray(datum["qi"]) / qi_unit,
            marker=".",
            linewidth=0.333,
            label=f"{labels[v['singular']]}" if not plotted[v["singular"]] else "",
            color=color,
        )
        plotted[v["singular"]] = True

    ff = FrozenFraction(
        volume=volume,
        droplet_volume=droplet_volume,
        total_particle_number=total_particle_number,
        rho_w=formulae.constants.rho_w,
    )
    twin = prim.secondary_yaxis(
        "right",
        functions=(lambda x: ff.qi2ff(x * qi_unit), lambda x: ff.ff2qi(x) / qi_unit),
    )
    twin.set_ylabel("frozen fraction")

    T = np.linspace(max(datum["T"]), min(datum["T"]))
    for multiplier, color in {0.1: "yellow", 1: "brown", 10: "orange"}.items():
        qi = (
            ff.ff2qi(
                formulae.freezing_temperature_spectrum.cdf(
                    T, multiplier * surf_spec.median
                )
            )
            / qi_unit
        )
        prim.plot(
            T,
            qi,
            label="" if multiplier != 1 else "singular CDFs for median surface",
            linewidth=2.5,
            color=color,
            linestyle="--",
        )
        if multiplier != 1:
            _ = CurvedText(
                x=T.squeeze(),
                y=qi.squeeze(),
                text=f"                      {multiplier}x median A",
                va="bottom",
                color="black",
                axes=prim,
            )
    title = f"$σ_g$=exp({np.log(surf_spec.s_geom):.3g})"
    if cooling_rate_K_min is not None:
        title += f", cooling rate: {cooling_rate_K_min} K/min"
    prim.set_title(title)
    # prim.set_ylabel('ice water content [$g/m^3$]')
    prim.set_yticks([])
    prim.set_xlim(T[0], T[-1])
    prim.legend(bbox_to_anchor=(1.02, -0.2))
    prim.grid()


def make_pdf_plot(A_spec, Shima_T_fz, A_range, T_range):
    N = 256
    T_space = np.linspace(*T_range, N)
    A_space = np.linspace(*A_range, N)
    grid = np.meshgrid(A_space, T_space)
    sampled_pdf = Shima_T_fz(grid[1], grid[0]) * A_spec.pdf(grid[0])

    fig = pyplot.figure(
        figsize=(4.5, 6),
    )
    ax = fig.add_subplot(111)
    ax.set_ylabel("freezing temperature [K]")
    ax.set_yticks(np.linspace(*T_range, num=5, endpoint=True))
    ax.set_xlabel("insoluble surface [μm$^2$]")

    data = sampled_pdf * si.um**2
    data[data == 0] = np.nan
    cnt = ax.contourf(
        grid[0] / si.um**2,
        grid[1],
        data,
        norm=matplotlib.colors.LogNorm(),
        cmap="Blues",
        levels=np.logspace(-3, 0, 7),
    )
    cbar = pyplot.colorbar(cnt, ticks=[0.001, 0.01, 0.1, 1.0], orientation="horizontal")
    cbar.set_label("pdf [K$^{-1}$ μm$^{-2}$]")
    ax.set_title(f"$σ_g$=exp({np.log(A_spec.s_geom):.3g})")

    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_histx.tick_params(axis="y", labelleft=False, left=False)
    ax_histy.tick_params(axis="y", labelleft=False, left=False)
    ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_histx.plot(A_space / si.um**2, np.sum(sampled_pdf, axis=0), color="teal")
    ax_histy.plot(np.sum(sampled_pdf, axis=1), T_space, color="black")

    pyplot.grid()
