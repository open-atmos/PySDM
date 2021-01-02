from ..conf import trtc
import numpy as np


class PrecisionResolver:
    _double_precision = False
    _conv_function = trtc.DVDouble if _double_precision else trtc.DVFloat
    _real_type = "double" if _double_precision else "float"
    _np_dtype = np.float64 if _double_precision else np.float32

    @staticmethod
    def get_floating_point(number):
        return PrecisionResolver._conv_function(number)

    # noinspection PyPep8Naming
    @staticmethod
    def get_C_type():
        return PrecisionResolver._real_type

    @staticmethod
    def get_np_dtype():
        return PrecisionResolver._np_dtype
