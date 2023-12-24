"""
Water isotopic line excess parameters used in
[Pierchala et al. 2022](https://10.1016/j.gca.2022.01.020)
"""

from .barkan_and_luz_2007 import BarkanAndLuz2007
from .dansgaard_1964 import Dansgaard1964


class PierchalaEtAl2022(Dansgaard1964, BarkanAndLuz2007):
    def __init__(self, _):
        pass
