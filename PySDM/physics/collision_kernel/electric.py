"""
kernel modelling influence of electric field of 3000V/cm
 as in [Berry 1967](https://doi.org/10.1175/1520-0469(1967)024%3C0688:CDGBC%3E2.0.CO;2)
"""

# TODO
# class Electric(Parameterized):  # pylint: disable=too-few-public-methods
#     def __init__(self):
#         super().__init__(
#             (1, 1, -7, 1.78, -20.5, 1.73, 0.26, 1.47, 1, 0.82, -0.003, 4.4, 8),
#         )

#
#     self.particulator.backend.linear_collection_efficiency(
#         params=self.params,
#         output=output,
#         radii=self.particulator.attributes["radius"],
#         is_first_in_pair=is_first_in_pair,
#         unit=const.si.um,
#     )
#     output **= 2
#     output *= const.PI
#     self.pair_tmp.max(self.particulator.attributes["radius"], is_first_in_pair)
#     self.pair_tmp **= 2
#     output *= self.pair_tmp
#
#     self.pair_tmp.distance(
#         self.particulator.attributes["relative fall velocity"], is_first_in_pair
#     )
#     output *= self.pair_tmp


class Electric:  # pylint:disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def requested_attributes(_):
        return "radius", "relative fall velocity", ""
