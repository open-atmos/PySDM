"""
do-nothing null formulation (needed as other formulations require parameters
 to be set before instantiation of Formulae)
"""


class Null:  # pylint: disable=unused-argument
    def __init__(self, _):
        pass

    @staticmethod
    def j_liq_homo(const, T, S, e_s):
        return 0

    @staticmethod
    def r_liq_homo(const, T, S):
        return 0
