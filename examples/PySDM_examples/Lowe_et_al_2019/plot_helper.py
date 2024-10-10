import numpy as np
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot


def plot_profiles(subplot_list, updraft_list, forg_list, output, save=False):
    _, axes = pyplot.subplots(
        len(subplot_list),
        len(updraft_list),
        sharex=False,
        sharey=True,
        figsize=(3 * len(updraft_list), 4 * len(subplot_list)),
    )

    for k, subplot in enumerate(subplot_list):
        for i, w in enumerate(updraft_list):
            for _, Forg in enumerate(forg_list):
                key = subplot + f"_w{w:.2f}_f{Forg:.2f}_"
                var = "CDNC_cm3"
                z = np.array(output[key + "CompressedFilmOvadnevaite"]["z"])
                CDNC_film = np.array(output[key + "CompressedFilmOvadnevaite"][var])
                CDNC_bulk = np.array(output[key + "Constant"][var])

                cmap = pyplot.get_cmap("Spectral")
                if len(subplot_list) > 1:
                    ax = axes[k, i]
                else:
                    ax = axes[i]

                ax.plot(CDNC_film, z, "--", color=cmap(Forg))
                ax.plot(CDNC_bulk, z, "-", color=cmap(Forg), label=f"{Forg:.2f}")

                if i == 0:
                    ax.set_ylabel("Parcel displacement [m]")
                    ax.set_title(subplot, loc="left", weight="bold")
                if i == len(updraft_list) - 1 and k == 0:
                    ax.legend(title="Forg", loc=2)
                if k == 0:
                    ax.set_title(f"w = {w:.2f} m/s")
                if k == len(subplot_list) - 1:
                    ax.set_xlabel("CDNC [cm$^{-3}$]")
    if save:
        show_plot("fig3_parcel_profiles.pdf")


def plot_contours(
    subplot_list, updraft_list, forg_list, output, actfrac=False, save=False
):
    _, axes = pyplot.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 10))

    for subplot in subplot_list:
        dCDNC = np.zeros((len(updraft_list), len(forg_list)))
        for i, w in enumerate(updraft_list):
            for j, Forg in enumerate(forg_list):
                key = subplot + f"_w{w:.2f}_f{Forg:.2f}_"
                if actfrac:
                    var = "Activated Fraction"
                    Naer = 1.0
                    CDNC_film = output[key + "CompressedFilmOvadnevaite"][var][0] * Naer
                    CDNC_bulk = output[key + "Constant"][var][0] * Naer
                else:
                    var = "CDNC_cm3"
                    z = np.array(output[key + "CompressedFilmOvadnevaite"]["z"])
                    wz = np.where(z == z[-1])[0][0]
                    CDNC_film = np.array(
                        output[key + "CompressedFilmOvadnevaite"][var]
                    )[wz]
                    CDNC_bulk = np.array(output[key + "Constant"][var])[wz]
                dCDNC[i, j] = (CDNC_film - CDNC_bulk) / CDNC_bulk * 100.0

        if subplot == "a":
            ax = axes[0, 0]
            ax.set_title(
                "MA Accum. mode conc. N$_2 = 30$ cm$^{-3}$", fontsize=13, loc="right"
            )
            ax.contour(
                forg_list,
                updraft_list,
                dCDNC,
                levels=[10, 25],
                colors=["#1fa8f2", "#4287f5"],
                linestyles=[":", "--"],
                linewidths=4,
            )
            p = ax.contourf(
                forg_list,
                updraft_list,
                dCDNC,
                cmap="Blues",
                levels=np.linspace(0, 90, 11),
                extend="both",
            )
        if subplot == "b":
            ax = axes[0, 1]
            ax.set_title(
                "MA Accum. mode conc. N$_2 = 135$ cm$^{-3}$", fontsize=13, loc="right"
            )
            ax.contour(
                forg_list,
                updraft_list,
                dCDNC,
                levels=[10, 25],
                colors=["#1fa8f2", "#4287f5"],
                linestyles=[":", "--"],
                linewidths=4,
            )
            p = ax.contourf(
                forg_list,
                updraft_list,
                dCDNC,
                cmap="Blues",
                levels=np.linspace(0, 90, 11),
                extend="both",
            )
        if subplot == "c":
            ax = axes[1, 0]
            ax.set_title(
                "HYY Accum. mode conc. N$_2 = 160$ cm$^{-3}$", fontsize=13, loc="right"
            )
            ax.contour(
                forg_list,
                updraft_list,
                dCDNC,
                levels=[10, 25],
                colors=["#04c753", "#157d3f"],
                linestyles=[":", "--"],
                linewidths=4,
            )
            p = ax.contourf(
                forg_list,
                updraft_list,
                dCDNC,
                cmap="Greens",
                levels=np.linspace(0, 75, 11),
                extend="both",
            )
        if subplot == "d":
            ax = axes[1, 1]
            ax.set_title(
                "HYY Accum. mode conc. N$_2 = 540$ cm$^{-3}$", fontsize=13, loc="right"
            )
            ax.contour(
                forg_list,
                updraft_list,
                dCDNC,
                levels=[10, 25],
                colors=["#04c753", "#157d3f"],
                linestyles=[":", "--"],
                linewidths=4,
            )
            p = ax.contourf(
                forg_list,
                updraft_list,
                dCDNC,
                cmap="Greens",
                levels=np.linspace(0, 75, 11),
                extend="both",
            )

        ax.set_title(subplot, weight="bold", loc="left")
        if subplot in ("c", "d"):
            ax.set_xlabel("Organic mass fraction")
        ax.set_yscale("log")
        ax.set_yticks([0.1, 1, 10])
        ax.set_yticklabels(["0.1", "1", "10"])
        ax.set_xlim([0, 1])
        if subplot in ("a", "c"):
            ax.set_ylabel("Updraft [ms$^{-1}$]")
        pyplot.colorbar(p, ax=ax, label=r"$\Delta_{CDNC}$ [%]")

    pyplot.rcParams.update({"font.size": 15})
    if save:
        if actfrac:
            pyplot.savefig(
                "fig_3_Scrit.png", dpi=200, facecolor="w", bbox_inches="tight"
            )
            show_plot("fig_3_Scrit.pdf")
        else:
            pyplot.savefig(
                "fig_3_rcrit.png", dpi=200, facecolor="w", bbox_inches="tight"
            )
            show_plot("fig_3_rcrit.pdf")
