from numpy import pi, log, exp


class VolumeLogarithm:
    @staticmethod
    def dx_dt(x, dr_dt):
        return exp(-x/3) * dr_dt * 3 * (3/4/pi)**(-1/3)

    @staticmethod
    def volume(x):
        return exp(x)

    @staticmethod
    def x(volume):
        return log(volume)