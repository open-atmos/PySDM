"""
for mixed-phase microphysics as in
[Shima et al. 2020](https://doi.org/10.5194/gmd-13-4107-2020)
"""


class PorousSpheroid:  # pylint: disable=too-few-public-methods
    @staticmethod
    def supports_mixed_phase(_=None):
        return True
