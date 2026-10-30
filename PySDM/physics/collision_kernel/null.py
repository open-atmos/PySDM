"""
do-nothing null formulation
"""


class Null:  # pylint: disable=unused-argument,too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def requested_attributes(_):
        return "", "", ""
