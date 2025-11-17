"""
groups of attributes used in either singular or time-dependent immersion freezing regimes,
in threshold or time-dependent homogeneous freezing and for thaw routines
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


class TimeDependentHomogeneousAttributes(
    namedtuple(
        typename="TimeDependentHomogeneousAttributes",
        field_names=("volume", "signed_water_mass"),
    )
):
    """groups attributes required in time-dependent regime for homogeneous freezing"""

    __slots__ = ()


class ThresholdHomogeneousAndThawAttributes(
    namedtuple(
        typename="ThresholdHomogeneousAttributes",
        field_names=("signed_water_mass"),
    )
):
    """groups attributes required in time-dependent regime for homogeneous freezing"""

    __slots__ = ()
