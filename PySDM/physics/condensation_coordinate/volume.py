from numpy import pi


class Volume:
    @staticmethod
    def dx_dt(x, dr_dt):
        r = radius(x)
        return 4 * pi * r**2 * dr_dt

    @staticmethod
    def volume(x):
        return x

    @staticmethod
    def x(volume):
        return volume
