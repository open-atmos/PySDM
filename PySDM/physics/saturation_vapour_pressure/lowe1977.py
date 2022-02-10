"""
polynomial fits from
[Lowe et al. 1977](https://www.jstor.org/stable/26177598)
"""
import numpy as np

class Lowe1977:
    def __init__(self, _):
        pass

    @staticmethod
    def pvs_Celsius(const, T):
        return (
                const.L77_A0 + T * (
                const.L77_A1 + T * (
                const.L77_A2 + T * (
                const.L77_A3 + T * (
                const.L77_A4 + T * (
                const.L77_A5 + T * (
                const.L77_A6
        )))))))

    @staticmethod
    def ice_Celsius(_, T):
        return np.nan * T
