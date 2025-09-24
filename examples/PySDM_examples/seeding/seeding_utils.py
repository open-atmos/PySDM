import numpy as np
import xarray
from tqdm import tqdm


def sav_as_xarr(field_name, settings, simulation):

    temp_arr = []
    for step in tqdm(settings.output_steps):
        temp_arr.append(simulation.storage.load(field_name, step))

    t_arr = np.linspace(
        0,
        settings.simulation_time * 10**9,
        int(settings.simulation_time / settings.output_interval) + 1,
        dtype="timedelta64[ns]",
    )
    min_bin_z = settings.size[0] / settings.grid[0] / 2
    min_bin_x = settings.size[1] / settings.grid[1] / 2
    z_arr = np.linspace(
        min_bin_z, settings.size[0] - min_bin_z, settings.grid[0], dtype=float
    )
    x_arr = np.linspace(
        min_bin_x, settings.size[1] - min_bin_x, settings.grid[1], dtype=float
    )

    temp_xarr = xarray.DataArray(
        data=np.transpose(np.array(temp_arr), (0, 2, 1)),
        dims=["T", "Z", "X"],
        coords=dict(
            T=(
                ["T"],
                t_arr,
            ),
            Z=(
                ["Z"],
                z_arr,
            ),
            X=(
                ["X"],
                x_arr,
            ),
        ),
    )

    return temp_xarr
