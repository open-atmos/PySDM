"""
integer-valued discretisation with sanity checks for errors due to type casting
"""

import numpy as np


def discretise_multiplicities(values_arg):
    """any NaN values in the input array are ignored and flagged
    with zero multiplicities in the output array"""

    values_int = np.where(np.isnan(values_arg), 0, values_arg).round().astype(np.int64)

    if np.issubdtype(values_arg.dtype, np.floating):
        if np.isnan(values_arg).all():
            return values_int

        if not np.logical_or(values_int > 0, np.isnan(values_arg)).all():
            raise ValueError(
                f"int-casting resulted in multiplicity of zero (min(y_float)={min(values_arg)})"
            )

        percent_diff = 100 * abs(
            1 - np.nansum(values_arg) / np.sum(values_int.astype(float))
        )
        if percent_diff > 1:
            raise ValueError(
                f"{percent_diff}% error in total real-droplet number"
                f" due to casting multiplicities to ints"
            )

    return values_int
