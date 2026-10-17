"""
TODO #744
"""

# TODO
# class Linear:
#     def __call__(self, output, is_first_in_pair):
#         output.sum_pair(self.particulator.attributes["volume"], is_first_in_pair)
#         output *= self.particulator.formulae.constants.LINEAR_b
#         output += self.particulator.formulae.constants.LINEAR_a


class Linear:  # pylint:disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def requested_attributes(_):
        return "", "", ""
