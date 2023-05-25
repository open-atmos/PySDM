# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Alpert_and_Knopf_2016 import Table1, simulation

from PySDM.physics import si

n_runs_per_case = 3


@pytest.mark.parametrize("multiplicity", (1, 2, 10))
def test_ak16_fig_1(multiplicity, plot=False):  # pylint: disable=too-many-locals
    # Arrange
    dt = 1 * si.s
    total_time = 6 * si.min

    # dummy multipliers (multiplied and then divided by)
    dv = (
        1 * si.cm**3
    )  # will become used if coalescence or other processes are turned on
    droplet_volume = 1 * si.um**3  # ditto

    cases = Table1(volume=dv)

    # Act
    output = {}

    for key in ("Iso3", "Iso4", "Iso1", "Iso2"):
        case = cases[key]
        output[key] = []
        for i in range(n_runs_per_case):
            number_of_real_droplets = case["ISA"].norm_factor * dv
            n_sd = number_of_real_droplets / multiplicity
            assert int(n_sd) == n_sd
            n_sd = int(n_sd)

            data, _ = simulation(
                constants={"J_HET": 1e3 / si.cm**2 / si.s},
                seed=i,
                n_sd=n_sd,
                time_step=dt,
                volume=dv,
                spectrum=case["ISA"],
                droplet_volume=droplet_volume,
                multiplicity=multiplicity,
                total_time=total_time,
                number_of_real_droplets=number_of_real_droplets,
            )
            output[key].append(data)

    # Plot
    for key, output_item in output.items():
        for run in range(n_runs_per_case):
            label = (
                f"{key}: σ=ln({int(cases[key]['ISA'].s_geom)}),"
                f"N={int(cases[key]['ISA'].norm_factor * dv)}"
            )
            pyplot.step(
                dt / si.min * np.arange(len(output_item[run])),
                output_item[run],
                label=label if run == 0 else None,
                color=cases[key]["color"],
                linewidth=0.666,
            )
        output_item.append(np.mean(np.asarray(output_item), axis=0))
        pyplot.step(
            dt / si.min * np.arange(len(output_item[-1])),
            output_item[-1],
            color=cases[key]["color"],
            linewidth=1.666,
        )

    pyplot.legend()
    pyplot.yscale("log")
    pyplot.ylim(1e-2, 1)
    pyplot.xlim(0, total_time / si.min)
    pyplot.xlabel("t / min")
    pyplot.ylabel("$f_{ufz}$")
    pyplot.gca().set_box_aspect(1)
    if plot:
        pyplot.show()

    # Assert
    np.testing.assert_array_less(
        output["Iso3"][-1][1 : int(1 * si.min / dt)],
        output["Iso1"][-1][1 : int(1 * si.min / dt)],
    )
    np.testing.assert_array_less(
        output["Iso1"][-1][int(2 * si.min / dt) :],
        output["Iso3"][-1][int(2 * si.min / dt) :],
    )
    np.testing.assert_array_less(
        output["Iso2"][int(0.5 * si.min / dt) :],
        output["Iso1"][int(0.5 * si.min / dt) :],
    )
    for out in output.values():
        np.testing.assert_array_less(1e-3, out[-1][: int(0.25 * si.min / dt)])
        np.testing.assert_array_less(out[-1][:], 1 + 1e-10)
