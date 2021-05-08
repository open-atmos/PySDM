from numpy import pi


class Volume:
    @staticmethod
    def dx_dt(x, r_dr_dt):
        return 4 * pi * (x * 3/4 / pi)**(1/3) * r_dr_dt

    @staticmethod
    def volume(x):
        return x

    @staticmethod
    def x(volume):
        return volume
