"""
do-nothing null formulation
"""


class Null:  # pylint: disable=unused-argument
    def __init__(self, _):
        pass

    @staticmethod
    def requested_parameters(_):
        return "", "", ""
