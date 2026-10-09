"""
disable liquid liquid collisions
"""


class Neglect:
    def __init__(self, _):
        pass

    @staticmethod
    def collision_kernel(const, output, is_first_in_pair):
        pass
