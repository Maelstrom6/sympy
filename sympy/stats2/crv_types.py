from sympy import *

class Normal(SingleContinuousDistribution):
    """
    A sample of how distributions will work.
    The user should be able to create their own distributions in the same way.
    You should be able to type as little as possible.

    It is up to the user to check that their PDF is nonnegative and integrates to 1 over its set.
    """
    set = Reals
    parameters = ('mu', 'sigma')

    def _check(self, mu, sigma):
        assert not (sigma <= 0)  # could be None

    def _pdf(self, x):
        mu = self.mu
        sigma = self.sigma
        return exp(((x-mu)/sigma)**2/2) / (sqrt(2*pi)*sigma)

    # or

    @property
    def _pdf(self) -> Lambda:
        mu = self.mu
        sigma = self.sigma
        x = self.symbol
        return Lambda(x, exp(((x - mu) / sigma) ** 2 / 2) / (sqrt(2 * pi) * sigma))

    # now Normal.pdf will return a Lambda of a Piecewise of _pdf over that set
    # cdf and all others will be computed on the fly
