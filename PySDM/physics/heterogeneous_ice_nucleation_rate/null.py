"""
do-nothing null formulation (needed as other formulations require parameters
 to be set before instantiation of Formulae)
"""


class Null:  # pylint: disable=too-few-public-methods,unused-argument
    def __init__(self, _):
        pass

    @staticmethod
    def j_het(const, a_w_ice):
        return 0
