"""
integer-valued discretisation with sanity checks for errors due to type casting
"""
import numpy as np


def discretise_n(y_float):
    y_int = y_float.round().astype(np.int64)

    percent_diff = 100 * abs(1 - np.sum(y_float) / np.sum(y_int.astype(float)))
    if percent_diff > 1:
        raise Exception(f"{percent_diff}% error in total real-droplet number due to casting multiplicities to ints")

    if not (y_int > 0).all():
        raise Exception("int-casting resulted in multiplicity of zero")

    return y_int
