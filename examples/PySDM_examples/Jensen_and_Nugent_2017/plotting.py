from matplotlib import pyplot
import numpy as np
from PySDM.physics import si, in_unit
from PySDM.physics.constants import PER_CENT


def figure(
    *,
    output,
    settings,
    simulation,
    plot_drops_with_dry_radii_um,
    xlim_r_um: tuple,
    xlim_S_percent: tuple,
):
    cloud_base = 300 * si.m
    y_axis = np.asarray(output["products"]["z"]) - settings.z0 - cloud_base

    masks = {}
    if settings.t_end_of_ascent is None:
        masks["ascent"] = np.full_like(output["products"]["t"], True, dtype=bool)
    else:
        masks["ascent"] = np.asarray(output["products"]["t"]) < settings.t_end_of_ascent
        masks["descent"] = np.logical_not(masks["ascent"])

    colors = {"ascent": "r", "descent": "b"}

    _, axs = pyplot.subplot_mosaic(
        mosaic=[["r", "S"]], width_ratios=[3, 1], sharey=True, tight_layout=True
    )

    for label, mask in masks.items():
        axs["S"].plot(
            in_unit(np.asarray(output["products"]["S_max"]), PER_CENT)[mask],
            y_axis[mask],
            label=label,
            color=colors[label],
        )
    axs["S"].set_xlim(*xlim_S_percent)
    axs["S"].set_xlabel("S (%)")
    axs["S"].legend()

    drop_ids = []
    for drop_size_um in plot_drops_with_dry_radii_um:
        drop_id = (np.abs(simulation.r_dry - drop_size_um * si.um)).argmin()
        drop_ids.append(drop_id)

    for (
        drop_id
    ) in (
        drop_ids
    ):  # TODO #1266: bug! why rightmost drop is not 500 nm if size range is set to end at 500 nm???
        for label, mask in masks.items():
            axs["r"].plot(
                in_unit(np.asarray(output["attributes"]["radius"][drop_id]), si.um)[
                    mask
                ],
                y_axis[mask],
                label=(
                    f"{in_unit(simulation.r_dry[drop_id], si.um):.2} µm"
                    if label == "ascent"
                    else ""
                ),
                color=colors[label],
            )
    axs["r"].legend()
    axs["r"].set_xlim(*xlim_r_um)
    axs["r"].set_xlabel("r$_c$ (µm)")
    axs["r"].set_ylabel("height above cloud base (m)")

    for ax in axs.values():
        ax.grid()
