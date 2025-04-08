"kinetic fractionation factor from [Merlivat and Jouzel 1979](https://doi.org/10.1029/JC084iC08p05029)"


class MerlivatAndJouzel1979:
    @staticmethod
    def alpha_kinetic(alpha_equilibrium, RH, D_heavy2D_light):
        return (
            alpha_equilibrium
            * RH
            / (alpha_equilibrium / D_heavy2D_light * (RH - 1) + 1)
        )
