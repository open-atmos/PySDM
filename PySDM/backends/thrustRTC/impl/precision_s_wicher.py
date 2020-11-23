from ..conf import trtc


class PrecisionResolver:
    _double_precision = True
    _conv_function = trtc.DVDouble if _double_precision else trtc.DVFloat
    _real_type = "double" if _double_precision else "float"

    @staticmethod
    def get_floating_point(number):
        return PrecisionResolver._conv_function(number)

    # noinspection PyPep8Naming
    @staticmethod
    def get_C_type():
        return PrecisionResolver._real_type
