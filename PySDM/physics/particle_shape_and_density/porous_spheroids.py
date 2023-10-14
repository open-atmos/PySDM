"""
for mixed-phase microphysics as in Shima et al. 2020
"""


class PorousSpheroid:  # pylint: disable=too-few-public-methods
    @staticmethod
    def supports_mixed_phase(_):
        return True
