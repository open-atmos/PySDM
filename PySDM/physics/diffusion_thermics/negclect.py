from PySDM.physics import constants as const


class Neglect:
    @staticmethod
    def D(T, p):
        return const.D0
