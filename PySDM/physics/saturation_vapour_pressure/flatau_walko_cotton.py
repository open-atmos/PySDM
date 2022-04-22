"""
polynomial fits from
[Flatau et al. 1992](https://doi.org/10.1175/1520-0450(1992)031%3C1507:PFTSVP%3E2.0.CO;2)
"""


class FlatauWalkoCotton:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        return const.FWC_C0 + T * (
            const.FWC_C1
            + T
            * (
                const.FWC_C2
                + T
                * (
                    const.FWC_C3
                    + T
                    * (
                        const.FWC_C4
                        + T
                        * (
                            const.FWC_C5
                            + T * (const.FWC_C6 + T * (const.FWC_C7 + T * const.FWC_C8))
                        )
                    )
                )
            )
        )

    @staticmethod
    def ice_Celsius(const, T):
        return const.FWC_I0 + T * (
            const.FWC_I1
            + T
            * (
                const.FWC_I2
                + T
                * (
                    const.FWC_I3
                    + T
                    * (
                        const.FWC_I4
                        + T
                        * (
                            const.FWC_I5
                            + T * (const.FWC_I6 + T * (const.FWC_I7 + T * const.FWC_I8))
                        )
                    )
                )
            )
        )
