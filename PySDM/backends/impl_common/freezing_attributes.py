from collections import namedtuple

SingularAttributes = namedtuple("_", ('freezing_temperature', 'wet_volume'))
TimeDependentAttributes = namedtuple("_", ('immersed_surface_area', 'wet_volume'))
