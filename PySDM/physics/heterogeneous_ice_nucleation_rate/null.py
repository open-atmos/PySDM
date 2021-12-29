"""
do-nothing null formulation (needed as other formulations require parameters
 to be set before instantiation of Formulae)
"""


class Null:
    def __init__(self, _):
        pass

    @staticmethod
    def j_het(const, a_w_ice):
        pass
