"""
Formulae supporting `PySDM.dynamics.collisions.breakup_fragmentations.lowlist82`
"""

import math

import numpy as np


class LowList1982Nf:  # pylint: disable=too-few-public-methods, too-many-locals
    def __init__(self, _):
        pass

    @staticmethod
    def params_f1(const, dl, dcoal):
        dcoalCM = dcoal / const.CM
        dlCM = dl / const.CM
        Hf1 = 50.8 * (dlCM) ** (-0.718)
        mu = dlCM
        sigma = 1 / Hf1
        for _ in range(10):
            sigma = (
                1
                / Hf1
                * np.sqrt(2 / np.pi)
                / (1 + math.erf((dcoalCM - dlCM) / (np.sqrt(2) * sigma)))
            )
        return (Hf1, mu, sigma)  # in cm

    @staticmethod
    def params_f2(const, ds):
        dsCM = ds / const.CM
        Hf2 = 4.18 * ((dsCM) ** (-1.17))
        mu = dsCM
        sigma = 1 / (np.sqrt(2 * np.pi) * Hf2)
        return (Hf2, mu, sigma)

    @staticmethod
    def params_f3(const, ds, dl):  # pylint: disable=too-many-locals
        dsCM = ds / const.CM
        dlCM = dl / const.CM
        # eq (3.3), (3.4)
        Ff1 = max(
            0,
            (
                (-2.25e4 * (dlCM - 0.403) ** 2 - 37.9) * (dsCM) ** (2.5)
                + 9.67 * (dlCM - 0.170) ** 2
                + 4.95
            ),
        )
        Ff2 = 1.02e4 * dsCM ** (2.83) + 2
        # eq (3.5)
        ds0 = max(0.04, (Ff1 / 2.83) ** (1 / 1.02e4))
        if dsCM > ds0:
            Ff = max(2.0, Ff1)
        else:
            Ff = max(2.0, Ff2)
        Dff3 = 0.241 * (dsCM) + 0.0129  # (4.14)
        # eq (4.18) - (4.21)
        Pf301 = 1.68e5 * dsCM ** (2.33)
        Pf302 = max(
            0,
            (
                (43.4 * (dlCM + 1.81) ** 2 - 159.0) / dsCM
                - 3870 * (dlCM - 0.285) ** 2
                - 58.1
            ),
        )
        alpha = (dsCM - ds0) / (0.2 * ds0)
        Pf303 = alpha * Pf301 + (1 - alpha) * Pf302
        if dsCM < ds0:
            Pf0 = Pf301
        elif dsCM > 1.2 * ds0:
            Pf0 = Pf302
        else:
            Pf0 = Pf303
        # eq (4.22), (4.16), (4.17) (4.23)
        sigmaf3 = 10 * Dff3
        muf3 = np.log(Dff3) + sigmaf3**2
        Hf3 = Pf0 * Dff3 / np.exp(-0.5 * sigmaf3**2)
        for _ in range(10):
            if sigmaf3 == 0.0 or Hf3 == 0:
                return (0.0, np.log(ds0), np.log(ds0))
            sigmaf3 = (
                np.sqrt(2 / np.pi)
                * (Ff - 2)
                / Hf3
                / (1 - math.erf((np.log(0.01) - muf3) / np.sqrt(2) / sigmaf3))
            )
            muf3 = np.log(Dff3) + sigmaf3**2
            Hf3 = Pf0 * Dff3 / np.exp(-0.5 * sigmaf3**2)

        return (Hf3, muf3, sigmaf3)

    @staticmethod
    def params_s1(const, dl, ds, dcoal):
        dsCM = ds / const.CM
        dlCM = dl / const.CM
        dcoalCM = dcoal / const.CM
        Hs1 = 100 * np.exp(-3.25 * dsCM)
        mus1 = dlCM
        sigmas1 = 1 / Hs1
        for _ in range(10):
            sigmas1 = (
                1
                / Hs1
                * np.sqrt(2 / np.pi)
                / (1 + math.erf((dcoalCM - dlCM) / (np.sqrt(2) * sigmas1)))
            )
        return (Hs1, mus1, sigmas1)  # in cm

    @staticmethod
    def params_s2(const, dl, ds, St):
        dsCM = ds / const.CM
        dlCM = dl / const.CM
        Dss2 = (
            0.254 * (dsCM ** (0.413)) * np.exp(3.53 * dsCM ** (2.51) * (dlCM - dsCM))
        )  # (4.27)
        bstar = 14.2 * np.exp(-17.2 * dsCM)
        Ps20 = 0.23 * dsCM ** (-3.93) * dlCM ** (bstar)  # (4.29)
        sigmas2 = 10 * Dss2  # as in (4.22)
        mus2 = np.log(Dss2) + sigmas2**2  # (4.32)
        Hs2 = Ps20 * Dss2 / np.exp(-0.5 * sigmas2**2)  # (4.28)

        Fs = 5 * math.erf((St - 2.52e-6) / (1.85e-6)) + 6  # (3.7)

        for _ in range(10):
            sigmas2 = (
                np.sqrt(2 / np.pi)
                * (Fs - 1)
                / Hs2
                / (1 - math.erf((np.log(0.01) - mus2) / np.sqrt(2) / sigmas2))
            )
            mus2 = np.log(Dss2) + sigmas2**2  # (4.32)
            Hs2 = Ps20 * Dss2 / np.exp(-0.5 * sigmas2**2)  # (4.28)

        return (Hs2, mus2, sigmas2)

    @staticmethod
    def params_d1(const, W1, dl, dcoal, CKE):
        dlCM = dl / const.CM
        dcoalCM = dcoal / const.CM
        mud1 = dlCM * (1 - np.exp(-3.70 * (3.10 - W1)))
        Hd1 = 1.58e-5 * CKE ** (-1.22)
        sigmad1 = 1 / Hd1
        for _ in range(10):
            sigmad1 = (
                1
                / Hd1
                * np.sqrt(2 / np.pi)
                / (1 + math.erf((dcoalCM - mud1) / (np.sqrt(2) * sigmad1)))
            )

        return (Hd1, mud1, sigmad1)  # in cm

    @staticmethod
    def params_d2(const, ds, dl, CKE):
        dsCM = ds / const.CM
        dlCM = dl / const.CM
        Ddd2 = np.exp(-17.4 * dsCM - 0.671 * (dlCM - dsCM)) * dsCM  # (4.37)
        bstar = 0.007 * dsCM ** (-2.54)  # (4.39)
        Pd20 = 0.0884 * dsCM ** (-2.52) * (dlCM - dsCM) ** (bstar)  # (4.38)
        sigmad2 = 10 * Ddd2

        mud2 = np.log(Ddd2) + sigmad2**2
        Hd2 = Pd20 * Ddd2 / np.exp(-0.5 * sigmad2**2)

        Fd = max(1.0, 297.5 + 23.7 * np.log(CKE))  # (3.9)
        if Fd == 1.0:
            return (0.0, np.log(Ddd2), np.log(Ddd2))

        for _ in range(10):
            if sigmad2 == 0.0 or Hd2 <= 0.1:
                return (0.0, np.log(Ddd2), np.log(Ddd2))
            if sigmad2 >= 1.0:
                return (0.0, np.log(Ddd2), np.log(Ddd2))
            sigmad2 = (
                np.sqrt(2 / np.pi)
                * (Fd - 1)
                / Hd2
                / (1 - math.erf((np.log(0.01) - mud2) / np.sqrt(2) / sigmad2))
            )
            mud2 = np.log(Ddd2) + sigmad2**2
            Hd2 = Pd20 * Ddd2 / np.exp(-0.5 * sigmad2**2)

        return (Hd2, mud2, sigmad2)
