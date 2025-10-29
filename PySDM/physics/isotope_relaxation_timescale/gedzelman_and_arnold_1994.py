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
    ):
        # pylint: disable=unused-argument
        return (
            R_liq
            / D_ratio_heavy_to_light
            * (relative_humidity - 1)
            * (relative_humidity + 1)
            / (1 + Fk * D_light)
            / (
                R_liq / alpha * (1 + Fk * D_light * relative_humidity)
                - R_vap * relative_humidity * (1 + Fk * D_light)
            )
        )
