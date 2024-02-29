from matplotlib import pyplot
import numpy as np
from PySDM.physics import si, in_unit
from PySDM.physics.constants import PER_CENT

CLOUD_BASE = 300 * si.m


def find_drop_ids_by_dry_size(plot_drops_with_dry_radii_um, simulation_r_dry):
    drop_ids = []
    for drop_size_um in plot_drops_with_dry_radii_um:
        drop_id = (np.abs(simulation_r_dry - drop_size_um * si.um)).argmin()
        drop_ids.append(drop_id)
    return drop_ids


def figure(
    *,
    output,
    settings,
    simulation,
    plot_drops_with_dry_radii_um,
    xlim_r_um: tuple,
    xlim_S_percent: tuple,
    return_masks: bool = False,
):
    y_axis = np.asarray(output["products"]["z"]) - settings.z0 - CLOUD_BASE

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

    drop_ids = find_drop_ids_by_dry_size(
        plot_drops_with_dry_radii_um=plot_drops_with_dry_radii_um,
        simulation_r_dry=simulation.r_dry,
    )

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

    if return_masks:
        return masks
