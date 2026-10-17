"""
basic geometric kernel
"""

#
# class Geometric(Gravitational):
#     def __call__(self, output, is_first_in_pair):
#         output.sum(self.particulator.attributes["radius"], is_first_in_pair)
#         output **= 2
#         output *= (
#             const.PI
#             * self.particulator.formulae.constants.GEOMETRIC_collection_efficiency
#         )
#         self.pair_tmp.distance(
#             self.particulator.attributes["relative fall velocity"], is_first_in_pair
#         )
#         output *= self.pair_tmp


class Geometric:  # pylint:disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def requested_attributes(_):
        return "", "", ""
