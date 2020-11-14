from ..conf import trtc


class PrecisionResolver:
    _double_prec = True
    _conv_function = trtc.DVDouble if _double_prec else trtc.DVFloat

    @staticmethod
    def get_floating_point(number):
        print("WESZLO!!!")
        return PrecisionResolver._conv_function(number)

