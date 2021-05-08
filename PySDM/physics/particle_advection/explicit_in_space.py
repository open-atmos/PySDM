class ExplicitInSpace:
    @staticmethod
    def displacement(omega, c_l, c_r):
        return c_l * (1 - omega) + c_r * omega
