"""
logic around `PySDM.initialisation.impl.spectrum.Spectrum` - parent class for most
 of the spectra
"""


class Spectrum:
    def __init__(self, distribution, distribution_params, norm_factor):
        self.distribution_params = distribution_params  # (loc, scale)
        self.norm_factor = norm_factor
        self.distribution = distribution

    def size_distribution(self, arg):
        result = self.norm_factor * self.distribution.pdf(
            arg, *self.distribution_params
        )
        return result

    def pdf(self, arg):
        return self.size_distribution(arg) / self.norm_factor

    def cdf(self, arg):
        return self.distribution.cdf(arg, *self.distribution_params)

    def stats(self, moments):
        result = self.distribution.stats(*self.distribution_params, moments)
        return result

    def cumulative(self, arg):
        result = self.norm_factor * self.distribution.cdf(
            arg, *self.distribution_params
        )
        return result

    def percentiles(self, cdf_values):
        result = self.distribution.ppf(cdf_values, *self.distribution_params)
        return result
