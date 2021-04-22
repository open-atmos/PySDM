from numpy import pi, log, exp


class VolumeLogarithm:
    @staticmethod
    def dx_dt(x, r_dr_dt):
        return exp(-2*x/3) * r_dr_dt * 3 * (3/4/pi)**(-2/3)

    @staticmethod
    def volume(x):
        return exp(x)

    @staticmethod
    def x(volume):
        return log(volume)