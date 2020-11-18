from ..conf import trtc


class PrecisionResolver:
    _double_precision = False
    _conv_function = trtc.DVDouble if _double_precision else trtc.DVFloat

    @staticmethod
    def get_floating_point(number):
        return PrecisionResolver._conv_function(number)
