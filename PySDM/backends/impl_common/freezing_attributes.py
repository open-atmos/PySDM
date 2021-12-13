from collections import namedtuple

__pdoc__ = {}

SingularAttributes = namedtuple("SingularAttributes", (
    'freezing_temperature',
    'wet_volume'
))
__pdoc__['SingularAttributes'] = False

TimeDependentAttributes = namedtuple("TimeDependentAttributes", (
    'immersed_surface_area',
    'wet_volume'
))
__pdoc__['TimeDependentAttributes'] = False
