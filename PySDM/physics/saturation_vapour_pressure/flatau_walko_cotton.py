from PySDM.physics import constants as const


class FlatauWalkoCotton:
    @staticmethod
    def pvs_Celsius(T):
        return (
                const.FWC_C0 + T * (
                const.FWC_C1 + T * (
                const.FWC_C2 + T * (
                const.FWC_C3 + T * (
                const.FWC_C4 + T * (
                const.FWC_C5 + T * (
                const.FWC_C6 + T * (
                const.FWC_C7 + T *
                const.FWC_C8
        ))))))))

    @staticmethod
    def ice_Celsius(T):
        return (
                const.FWC_I0 + T * (
                const.FWC_I1 + T * (
                const.FWC_I2 + T * (
                const.FWC_I3 + T * (
                const.FWC_I4 + T * (
                const.FWC_I5 + T * (
                const.FWC_I6 + T * (
                const.FWC_I7 + T *
                const.FWC_I8
        ))))))))

    @staticmethod
    def a_w_ice(T):
        return (
            FlatauWalkoCotton.ice_Celsius(T - const.T0)
            /
            FlatauWalkoCotton.pvs_Celsius(T - const.T0)
        )
