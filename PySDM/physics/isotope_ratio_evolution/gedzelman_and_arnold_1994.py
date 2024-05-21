"""
eqs. (22) and (23) in [Gedzelman & Arnold 1994](https://doi.org/10.1029/93JD03518)
"""


# pylint: disable=too-few-public-methods
class GedzelmanAndArnold1994:
    def __init__(self, _):
        pass

    # pylint: disable=too-many-arguments
    @staticmethod
    def zero_dR_condition(
        _, diff_rat, iso_ratio_x, iso_ratio_r, iso_ratio_v, b, alpha_w
    ):
        return (diff_rat * iso_ratio_x - iso_ratio_r / alpha_w) / (
            diff_rat * iso_ratio_x - (1 + b) * iso_ratio_v + b * iso_ratio_r / alpha_w
        )
