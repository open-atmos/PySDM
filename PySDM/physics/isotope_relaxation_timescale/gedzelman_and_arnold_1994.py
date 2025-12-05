class GedzelmanAndArnold1994:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def bolin_number(
        const,
        D_ratio_heavy_to_light,
        alpha,
        D_light,
        Fk,
        R_vap,
        R_liq,
        relative_humidity,
        rho_v,
    ):
        """from eq. 20 and eq. 16 we corrected eq. 17 and calculate Bolin number"""
        # pylint: disable=unused-argument
        missing_factor_b = rho_v
        b = Fk * D_light
        return (
            alpha
            / D_ratio_heavy_to_light
            * (1 - relative_humidity)
            / (1 + missing_factor_b * b)
            / (
                relative_humidity * (1 - R_vap * alpha / R_liq)
                + (1 - relative_humidity) / (1 + missing_factor_b * b)
            )
        )
