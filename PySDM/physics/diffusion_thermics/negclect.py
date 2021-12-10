from PySDM.physics import constants as const


class Neglect:
    @staticmethod
    def D(T, p):  # pylint: disable=unused-argument
        return const.D0
