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
