"""
CPU implementation of backend methods supporting fragmentation functions
"""

from functools import cached_property
import numba
import numpy as np
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_common.backend_methods import BackendMethods


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_Nr(  # pylint: disable=too-many-arguments,unused-argument
    i,
    Nr1,
    Nr2,
    Nr3,
    Nr4,
    Nrt,
    CW,
    gam,
):  # pylint: disable=too-many-branches`
    if gam[i] * CW[i] >= 7.0:
        Nr1[i] = 0.088 * (gam[i] * CW[i] - 7.0)
    if CW[i] >= 21.0:
        Nr2[i] = 0.22 * (CW[i] - 21.0)
        if CW[i] <= 46.0:
            Nr3[i] = 0.04 * (46.0 - CW[i])
    else:
        Nr3[i] = 1.0
    Nr4[i] = 1.0
    Nrt[i] = Nr1[i] + Nr2[i] + Nr3[i] + Nr4[i]


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_mass_remainder(  # pylint: disable=too-many-arguments,unused-argument
    i, vl, ds, mu1, sigma1, mu2, sigma2, mu3, sigma3, d34, Nr1, Nr2, Nr3, Nr4
):
    # pylint: disable=too-many-arguments, too-many-locals
    Nr1[i] = Nr1[i] * np.exp(3 * mu1 + 9 * np.power(sigma1, 2) / 2)
    Nr2[i] = Nr2[i] * (mu2**3 + 3 * mu2 * sigma2**2)
    Nr3[i] = Nr3[i] * (mu3**3 + 3 * mu3 * sigma3**2)
    Nr4[i] = vl[i] * 6 / np.pi + ds[i] ** 3 - Nr1[i] - Nr2[i] - Nr3[i]
    if Nr4[i] <= 0.0:
        d34[i] = 0
        Nr4[i] = 0
    else:
        d34[i] = np.exp(np.log(Nr4[i]) / 3)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def ll82_Nr(  # pylint: disable=too-many-arguments,unused-argument
    i,
    Rf,
    Rs,
    Rd,
    CKE,
    W,
    W2,
):  # pylint: disable=too-many-branches`
    if CKE[i] >= 0.893e-6:
        Rf[i] = 1.11e-4 * CKE[i] ** (-0.654)
    else:
        Rf[i] = 1.0
    if W[i] >= 0.86:
        Rs[i] = 0.685 * (1 - np.exp(-1.63 * (W2[i] - 0.86)))
    else:
        Rs[i] = 0.0
    if (Rs[i] + Rf[i]) > 1.0:
        Rd[i] = 0.0
    else:
        Rd[i] = 1.0 - Rs[i] - Rf[i]


class FragmentationMethods(BackendMethods):
    @cached_property
    def _fragmentation_limiters_body(self):
        @numba.njit(**self.default_jit_flags)
        # pylint: disable=too-many-arguments
        def body(n_fragment, frag_volume, vmin, nfmax, x_plus_y):
            for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
                if x_plus_y[i] == 0.0:
                    frag_volume[i] = 0.0
                    n_fragment[i] = 1.0
                else:
                    if np.isnan(frag_volume[i]) or frag_volume[i] == 0.0:
                        frag_volume[i] = x_plus_y[i]
                    frag_volume[i] = min(frag_volume[i], x_plus_y[i])
                    if nfmax is not None and x_plus_y[i] / frag_volume[i] > nfmax:
                        frag_volume[i] = x_plus_y[i] / nfmax
                    elif frag_volume[i] < vmin:
                        frag_volume[i] = x_plus_y[i]
                    n_fragment[i] = x_plus_y[i] / frag_volume[i]

        return body

    def fragmentation_limiters(self, *, n_fragment, frag_volume, vmin, nfmax, x_plus_y):
        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            vmin=vmin,
            nfmax=nfmax,
            x_plus_y=x_plus_y.data,
        )

    @cached_property
    def _slams_fragmentation_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(n_fragment, frag_volume, x_plus_y, probs, rand):
            for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
                probs[i] = 0.0
                n_fragment[i] = 1
                for n in range(22):
                    probs[i] += 0.91 * (n + 2) ** (-1.56)
                    if rand[i] < probs[i]:
                        n_fragment[i] = n + 2
                        break
                frag_volume[i] = x_plus_y[i] / n_fragment[i]

        return body

    def slams_fragmentation(
        self, n_fragment, frag_volume, x_plus_y, probs, rand, vmin, nfmax
    ):  # pylint: disable=too-many-arguments
        self._slams_fragmentation_body(
            n_fragment.data, frag_volume.data, x_plus_y.data, probs.data, rand.data
        )
        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            vmin=vmin,
            nfmax=nfmax,
            x_plus_y=x_plus_y.data,
        )

    @cached_property
    def _exp_fragmentation_body(self):
        @numba.njit(**self.default_jit_flags)
        # pylint: disable=too-many-arguments
        def body(*, scale, frag_volume, rand, tol=1e-5):
            for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
                frag_volume[i] = -scale * np.log(max(1 - rand[i], tol))

        return body

    def exp_fragmentation(
        self,
        *,
        n_fragment,
        scale,
        frag_volume,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        tol=1e-5,
    ):
        self._exp_fragmentation_body(
            scale=scale,
            frag_volume=frag_volume.data,
            rand=rand.data,
            tol=tol,
        )
        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def feingold1988_fragmentation(
        self,
        *,
        n_fragment,
        scale,
        frag_volume,
        x_plus_y,
        rand,
        fragtol,
        vmin,
        nfmax,
    ):
        self._feingold1988_fragmentation_body(
            scale=scale,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            rand=rand.data,
            fragtol=fragtol,
        )

        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def gauss_fragmentation(
        self, *, n_fragment, mu, sigma, frag_volume, x_plus_y, rand, vmin, nfmax
    ):
        self._gauss_fragmentation_body(
            mu=mu,
            sigma=sigma,
            frag_volume=frag_volume.data,
            rand=rand.data,
        )
        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def straub_fragmentation(
        # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        n_fragment,
        CW,
        gam,
        ds,
        frag_volume,
        v_max,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        Nr1,
        Nr2,
        Nr3,
        Nr4,
        Nrt,
        d34,
    ):
        self._straub_fragmentation_body(
            CW=CW.data,
            gam=gam.data,
            ds=ds.data,
            frag_volume=frag_volume.data,
            v_max=v_max.data,
            rand=rand.data,
            Nr1=Nr1.data,
            Nr2=Nr2.data,
            Nr3=Nr3.data,
            Nr4=Nr4.data,
            Nrt=Nrt.data,
            d34=d34.data,
        )
        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def ll82_fragmentation(
        # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        n_fragment,
        CKE,
        W,
        W2,
        St,
        ds,
        dl,
        dcoal,
        frag_volume,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        Rf,
        Rs,
        Rd,
        tol=1e-8,
    ):
        self._ll82_fragmentation_body(
            CKE=CKE.data,
            W=W.data,
            W2=W2.data,
            St=St.data,
            ds=ds.data,
            dl=dl.data,
            dcoal=dcoal.data,
            frag_volume=frag_volume.data,
            rand=rand.data,
            Rf=Rf.data,
            Rs=Rs.data,
            Rd=Rd.data,
            tol=tol,
        )
        self._fragmentation_limiters_body(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    @cached_property
    def _ll82_coalescence_check_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(*, Ec, dl):
            for i in numba.prange(len(Ec)):  # pylint: disable=not-an-iterable
                if dl[i] < 0.4e-3:
                    Ec[i] = 1.0

        return body

    def ll82_coalescence_check(self, *, Ec, dl):
        self._ll82_coalescence_check_body(
            Ec=Ec.data,
            dl=dl.data,
        )

    @cached_property
    def _straub_fragmentation_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(
            *, CW, gam, ds, v_max, frag_volume, rand, Nr1, Nr2, Nr3, Nr4, Nrt, d34
        ):  # pylint: disable=too-many-arguments,too-many-locals
            for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
                straub_Nr(i, Nr1, Nr2, Nr3, Nr4, Nrt, CW, gam)
                sigma1 = ff.fragmentation_function__params_sigma1(CW[i])
                mu1 = ff.fragmentation_function__params_mu1(sigma1)
                sigma2 = ff.fragmentation_function__params_sigma2(CW[i])
                mu2 = ff.fragmentation_function__params_mu2(ds[i])
                sigma3 = ff.fragmentation_function__params_sigma3(CW[i])
                mu3 = ff.fragmentation_function__params_mu3(ds[i])
                straub_mass_remainder(
                    i,
                    v_max,
                    ds,
                    mu1,
                    sigma1,
                    mu2,
                    sigma2,
                    mu3,
                    sigma3,
                    d34,
                    Nr1,
                    Nr2,
                    Nr3,
                    Nr4,
                )
                Nrt[i] = Nr1[i] + Nr2[i] + Nr3[i] + Nr4[i]

                if Nrt[i] == 0.0:
                    diameter = 0.0
                else:
                    if rand[i] < Nr1[i] / Nrt[i]:
                        X = rand[i] * Nrt[i] / Nr1[i]
                        lnarg = mu1 + np.sqrt(2) * sigma1 * ff.trivia__erfinv_approx(X)
                        diameter = np.exp(lnarg)
                    elif rand[i] < (Nr2[i] + Nr1[i]) / Nrt[i]:
                        X = (rand[i] * Nrt[i] - Nr1[i]) / Nr2[i]
                        diameter = mu2 + np.sqrt(2) * sigma2 * ff.trivia__erfinv_approx(
                            X
                        )
                    elif rand[i] < (Nr3[i] + Nr2[i] + Nr1[i]) / Nrt[i]:
                        X = (rand[i] * Nrt[i] - Nr1[i] - Nr2[i]) / Nr3[i]
                        diameter = mu3 + np.sqrt(2) * sigma3 * ff.trivia__erfinv_approx(
                            X
                        )
                    else:
                        diameter = d34[i]

                frag_volume[i] = diameter**3 * ff.constants.PI / 6

        return body

    @cached_property
    def _ll82_fragmentation_body(self):  # pylint: disable=too-many-statements
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(
            *, CKE, W, W2, St, ds, dl, dcoal, frag_volume, rand, Rf, Rs, Rd, tol
        ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
            for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
                if dl[i] <= 0.4e-3:
                    frag_volume[i] = dcoal[i] ** 3 * ff.constants.PI / 6
                elif ds[i] == 0.0 or dl[i] == 0.0:
                    frag_volume[i] = 1e-18
                else:
                    ll82_Nr(i, Rf, Rs, Rd, CKE, W, W2)
                    if rand[i] <= Rf[i]:  # filament breakup
                        (H1, mu1, sigma1) = ff.fragmentation_function__params_f1(
                            dl[i], dcoal[i]
                        )
                        (H2, mu2, sigma2) = ff.fragmentation_function__params_f2(ds[i])
                        (H3, mu3, sigma3) = ff.fragmentation_function__params_f3(
                            ds[i], dl[i]
                        )
                        H1 = H1 * mu1
                        H2 = H2 * mu2
                        H3 = H3 * np.exp(mu3)
                        Hsum = H1 + H2 + H3
                        rand[i] = rand[i] / Rf[i]
                        if rand[i] <= H1 / Hsum:
                            X = max(rand[i] * Hsum / H1, tol)
                            frag_volume[i] = mu1 + np.sqrt(
                                2
                            ) * sigma1 * ff.trivia__erfinv_approx(2 * X - 1)
                        elif rand[i] <= (H1 + H2) / Hsum:
                            X = (rand[i] * Hsum - H1) / H2
                            frag_volume[i] = mu2 + np.sqrt(
                                2
                            ) * sigma2 * ff.trivia__erfinv_approx(2 * X - 1)
                        else:
                            X = min((rand[i] * Hsum - H1 - H2) / H3, 1.0 - tol)
                            lnarg = mu3 + np.sqrt(
                                2
                            ) * sigma3 * ff.trivia__erfinv_approx(2 * X - 1)
                            frag_volume[i] = np.exp(lnarg)

                    elif rand[i] <= Rf[i] + Rs[i]:  # sheet breakup
                        (H1, mu1, sigma1) = ff.fragmentation_function__params_s1(
                            dl[i], ds[i], dcoal[i]
                        )
                        (H2, mu2, sigma2) = ff.fragmentation_function__params_s2(
                            dl[i], ds[i], St[i]
                        )
                        H1 = H1 * mu1
                        H2 = H2 * np.exp(mu2)
                        Hsum = H1 + H2
                        rand[i] = (rand[i] - Rf[i]) / (Rs[i])
                        if rand[i] <= H1 / Hsum:
                            X = max(rand[i] * Hsum / H1, tol)
                            frag_volume[i] = mu1 + np.sqrt(
                                2
                            ) * sigma1 * ff.trivia__erfinv_approx(2 * X - 1)
                        else:
                            X = min((rand[i] * Hsum - H1) / H2, 1.0 - tol)
                            lnarg = mu2 + np.sqrt(
                                2
                            ) * sigma2 * ff.trivia__erfinv_approx(2 * X - 1)
                            frag_volume[i] = np.exp(lnarg)

                    else:  # disk breakup
                        (H1, mu1, sigma1) = ff.fragmentation_function__params_d1(
                            W[i], dl[i], dcoal[i], CKE[i]
                        )
                        (H2, mu2, sigma2) = ff.fragmentation_function__params_d2(
                            ds[i], dl[i], CKE[i]
                        )
                        H1 = H1 * mu1
                        Hsum = H1 + H2
                        rand[i] = (rand[i] - Rf[i] - Rs[i]) / Rd[i]
                        if rand[i] <= H1 / Hsum:
                            X = max(rand[i] * Hsum / H1, tol)
                            frag_volume[i] = mu1 + np.sqrt(
                                2
                            ) * sigma1 * ff.trivia__erfinv_approx(2 * X - 1)
                        else:
                            X = min((rand[i] * Hsum - H1) / H2, 1 - tol)
                            lnarg = mu2 + np.sqrt(
                                2
                            ) * sigma2 * ff.trivia__erfinv_approx(2 * X - 1)
                            frag_volume[i] = np.exp(lnarg)

                    frag_volume[i] = (
                        frag_volume[i] * 0.01
                    )  # diameter in cm; convert to m
                    frag_volume[i] = frag_volume[i] ** 3 * ff.constants.PI / 6

        return body

    @cached_property
    def _gauss_fragmentation_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(*, mu, sigma, frag_volume, rand):  # pylint: disable=too-many-arguments
            for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
                frag_volume[i] = mu + sigma * ff.trivia__erfinv_approx(rand[i])

        return body

    @cached_property
    def _feingold1988_fragmentation_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        # pylint: disable=too-many-arguments
        def body(*, scale, frag_volume, x_plus_y, rand, fragtol):
            for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
                frag_volume[i] = ff.fragmentation_function__frag_volume(
                    scale, rand[i], x_plus_y[i], fragtol
                )

        return body
