"""
integer-valued discretisation with sanity checks for errors due to type casting
"""

import numpy as np


def discretise_multiplicities(values_arg):
    values_int = values_arg.round().astype(np.int64)

    if np.issubdtype(values_arg.dtype, np.floating):
        percent_diff = 100 * abs(
            1 - np.sum(values_arg) / np.sum(values_int.astype(float))
        )
        if percent_diff > 1:
            raise ValueError(
                f"{percent_diff}% error in total real-droplet number"
                f" due to casting multiplicities to ints"
            )

        if not (values_int > 0).all():
            raise ValueError(
                f"int-casting resulted in multiplicity of zero (min(y_float)={min(values_arg)})"
            )

    return values_int
