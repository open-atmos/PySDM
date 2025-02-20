"""
groups of attributes used in either singular or time-dependent immersion freezing regimes
"""

from collections import namedtuple


class SingularAttributes(
    namedtuple(
        typename="SingularAttributes",
        field_names=("freezing_temperature", "signed_water_mass"),
    )
):
    """groups attributes required in singular regime"""

    __slots__ = ()


class TimeDependentAttributes(
    namedtuple(
        typename="TimeDependentAttributes",
        field_names=("immersed_surface_area", "signed_water_mass"),
    )
):
    """groups attributes required in time-dependent regime"""

    __slots__ = ()
