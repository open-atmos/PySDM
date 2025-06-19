"""
eqs. 14-16 in [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015)
"""
import numpy as np

class ImplicitInSpace:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def displacement(_, position_in_cell, cell_id, c_l, c_r, enable_monte_carlo, u01):
        return (
            position_in_cell
            + (
                np.floor(
                    np.abs(max(c_l, c_r))
                )
                * np.sign(
                    np.abs(max(c_l, c_r))
                    / max(c_l, c_r)
                )
            )
            + (
                np.abs(max(c_l, c_r)) > u01
            )
            * np.sign(
                np.abs(max(c_l, c_r))
                / max(c_l, c_r)
            )
        ) \
        if enable_monte_carlo \
        else (
            (
                c_l
                * (1 - position_in_cell)
                + c_r
                * position_in_cell
            )
            / (
                1
                - c_r
                + c_l
            )
        )
