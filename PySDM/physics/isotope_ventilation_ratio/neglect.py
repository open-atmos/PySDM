"""assuming f'/f equals 1"""


class Neglect:  # pylint disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def isotope_ventilation_ratio_heavy_to_light(
        ventilation_coefficient, diffusivity_ratio
    ):  # pylint: disable=unused-argument
        return 1
