from numpy import pi


class Volume:
    @staticmethod
    def dx_dt(x, dr_dt):
        return 4 * pi * (x * 3/4 / pi)**(2/3) * dr_dt

    @staticmethod
    def volume(x):
        return x

    @staticmethod
    def x(volume):
        return volume
