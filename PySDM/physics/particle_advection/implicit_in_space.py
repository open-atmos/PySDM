class ImplicitInSpace:
    # eqs. 14-16 in Arabas et al. 2015 (libcloudph++)
    @staticmethod
    def displacement(omega, c_l, c_r):
        return (omega * c_r + c_l * (1 - omega)) / (1 - c_r + c_l)
