"""
polynomial fits from
[Lowe et al. 1977](https://doi.org/10.1175/1520-0450(1977)016<0100:AAPFTC>2.0.CO;2)
"""


class Lowe1977:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        return const.L77W_A0 + T * (
            const.L77W_A1
            + T
            * (
                const.L77W_A2
                + T
                * (
                    const.L77W_A3
                    + T * (const.L77W_A4 + T * (const.L77W_A5 + T * (const.L77W_A6)))
                )
            )
        )

    @staticmethod
    def ice_Celsius(const, T):
        return const.L77I_A0 + T * (
            const.L77I_A1
            + T
            * (
                const.L77I_A2
                + T
                * (
                    const.L77I_A3
                    + T * (const.L77I_A4 + T * (const.L77I_A5 + T * (const.L77I_A6)))
                )
            )
        )
