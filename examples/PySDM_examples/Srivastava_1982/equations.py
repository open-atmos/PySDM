import numpy as np


class Equations:
    """Equations from Srivastava 1982: "A Simple Model of Particle Coalescence and Breakup"
    (https://doi.org/10.1175/1520-0469(1982)039%3C1317:ASMOPC%3E2.0.CO;2)
    note: all equations assume constant fragment mass"""

    @property
    def alpha_star(self):
        """see eq. 6"""
        return self._alpha_star or self.alpha / self.c / self.M

    @property
    def beta_star(self):
        """see eq. 6"""
        return self._beta_star or self.beta / self.c

    def tau(self, t):
        """see eq. 6"""
        return self.c * self.M * t

    def __init__(
        self, *, M=None, c=None, alpha=None, beta=None, alpha_star=None, beta_star=None
    ):
        if alpha_star and (alpha or M or c):
            raise ValueError("conflicting parameter")
        self.M = M
        self.c = c
        self.alpha = alpha
        self.beta = beta

        self._beta_star = beta_star
        self._alpha_star = alpha_star

        if alpha_star and beta_star:
            amb = alpha_star - beta_star
            self._A = amb / 2 / alpha_star
            self._B = 1 / np.sqrt(
                (0.5 / alpha_star + beta_star / alpha_star)
                + amb**2 / (4 * alpha_star**2)
            )

    def eq12(self):
        """equilibrium (τ→∞) mean mass under collisions and spontaneous breakup
        (no collisional breakup)
        expressed as a ratio to fragment mass (i.e., dimensionless)"""
        equilibrium_mean_mass_to_frag_mass_ratio = (
            0.5 + (0.25 + 0.5 / self.alpha_star) ** 0.5
        )
        return equilibrium_mean_mass_to_frag_mass_ratio

    def eq13(self, m0, tau):
        """mean mass expressed as a ratio to fragment mass as a function of
        dimensionless scaled time (τ) under coalescence and collisional breakup
        (no spontaneous breakup)"""
        mean_mass_to_frag_mass_ratio = self._eq13(m0, tau)
        return mean_mass_to_frag_mass_ratio

    def _eq13(self, m0, tau):
        ebt = np.exp(-self.beta_star * tau)
        return m0 * ebt + (1 + 0.5 / self.beta_star) * (1 - ebt)

    def eq14(self):
        """equilibrium (τ→∞) mean mass expressed as a ratio to fragment mass for
        under collisional merging and breakup (no spontaneous breakup)"""
        equilibrium_mean_mass_to_frag_mass_ratio = 1 + 0.5 / self.beta_star
        return equilibrium_mean_mass_to_frag_mass_ratio

    def eq15(self, m):
        return (m - self._A) * self._B

    def eq15_m_of_y(self, y):
        return (y / self._B) + self._A

    def eq16(self, tau):
        return tau * self.alpha_star / self._B

    def eq10(self, m0, tau):
        """ratio of mean mass to fragment size mass as a function of scaled time
        for the case of coalescence only"""
        mean_mass_to_frag_mass_ratio = m0 + tau / 2
        return mean_mass_to_frag_mass_ratio


class EquationsHelpers:
    def __init__(self, total_volume, total_number_0, rho, frag_mass):
        self.total_volume = total_volume
        self.total_number_0 = total_number_0
        self.rho = rho
        self.frag_mass = frag_mass

    def m0(self):
        mean_volume_0 = self.total_volume / self.total_number_0
        m0 = self.rho * mean_volume_0 / self.frag_mass
        return m0
