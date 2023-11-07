import sys
import importlib.util
import subprocess

if __name__ != '__main__':
    if importlib.util.find_spec('scipy') is None:
        i = input("[distributions.py] Package 'scipy' is required. Install it? [Y/N] ")
        if i == 'Y' or i == 'y':
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', 'scipy'])
            print('Done. Now, re-run the script.')
            exit(0)


from typing import Callable
from scipy.special import erf, gamma, hyp2f1, gammainc, beta, betainc
import math


class ContinuousDistribution:
    """
    This represents a continuous distribution on the real numbers.
    """

    def __init__(self, pdf: Callable[[float], float], cdf: Callable[[float], float], name: str = ''):
        self.__pdf = pdf
        self.__cdf = cdf
        self.__name = name

    def get_pdf_at(self, x: float) -> float:
        return self.__pdf(x)

    def get_cdf_at(self, x: float) -> float:
        return self.__cdf(x)

    def get_prob_between(self, a: float, b: float) -> float:
        """
        Returns the value of P(a <= X <= b).
        """
        return self.__cdf(b) - self.__cdf(a)

    def get_name(self) -> str:
        return self.__name

    def get_quantile(self, x: float, guess: float = .5, rep: int = 1000) -> float:
        """
        Calculate the inverse function of cdf at x.
        In other words, get_cdf_at(get_quantile(x)) == x.

        :param guess: The initial guess of the inverse function of cdf at x.
        :param rep: The number of repetitions. The higher `rep` is, the more accurate the result is.
        """
        assert 0 < x < 1, "x must be between 0 and 1."
        res = guess
        for _ in range(rep):
            res -= (self.__cdf(res) - x) / self.__pdf(res)
        return res

    def get_score(self, x: float, guess: float = .5, rep: int = 1000) -> float:
        """
        Calculate the value z such that cdf(z) = 1-x, i.e., P(X > z) = x.
        This is essentially get_quantile(1-x).

        :param guess: The initial guess of the inverse function of cdf at x.
        :param rep: The number of repetitions. The higher `rep` is, the more accurate the result is.
        """
        return self.get_quantile(1 - x, guess, rep)

    def __repr__(self):
        return f'<Distribution {self.__name}>'


def hyper_geometric(a: float, b: float, c: float, z: float, rep: int = 0) -> float:
    if rep <= 0:
        rep = max(int(math.ceil(a * b * z / c) * 10), 100)
        # rep = math.ceil(a * b / c)
    res = coeff = 1.
    for i in range(1, rep + 1):
        coeff *= a * b * z / c / i
        # print(coeff, res)
        res += coeff
        a += 1
        b += 1
        c += 1
    return res


def normal_distribution(mu: float, sigma_squared: float) -> ContinuousDistribution:
    assert sigma_squared > 0, "sigma_squared must be positive."
    sigma = sigma_squared ** .5

    def pdf(x):
        return math.exp(-.5 * ((x - mu)/sigma) ** 2) / (sigma * (math.tau) ** .5)

    def cdf(x):
        return .5 * (1 + erf((x - mu) / (sigma * 2 ** .5)))
    return ContinuousDistribution(pdf, cdf, f'Normal({mu:.4g}, {sigma_squared:.4g})')


def gamma_distribution(alpha: float, l: float) -> ContinuousDistribution:
    assert alpha > 0, "alpha must be positive."
    assert l > 0, "l must be positive."

    def pdf(x):
        if x < 0:
            return 0
        lx = l * x
        return l * math.exp(-lx) * lx ** (alpha - 1) / gamma(alpha)

    def cdf(x):
        return gammainc(alpha, l * x)
    return ContinuousDistribution(pdf, cdf, f'Gamma({alpha:.4g}, {l:.4g})')


def chi_squared_distribution(k: int) -> ContinuousDistribution:
    assert int(k) == k and k > 0, "k must be a positive integer."
    alpha = k / 2
    l = .5

    def pdf(x):
        if x < 0:
            return 0
        lx = l * x
        return l * math.exp(-lx) * lx ** (alpha - 1) / gamma(alpha)

    def cdf(x):
        return gammainc(alpha, l * x)
    return ContinuousDistribution(pdf, cdf, f'ChiSquared({k})')


def student_t_distribution(nu: float) -> ContinuousDistribution:
    assert nu > 0, "nu must be positive."
    coeff = math.gamma((nu + 1) / 2) / ((nu * math.pi)
                                        ** .5 * math.gamma(nu / 2))

    def pdf(x):
        return coeff * (1 + x**2 / nu) ** (-(nu + 1)/2)

    def cdf(x):
        return .5 + x * coeff * hyp2f1(.5, (nu + 1) / 2, 1.5, -x**2 / nu)

    return ContinuousDistribution(pdf, cdf, f'Student T({nu:.4g})')


def f_distribution(n: float, m: float) -> ContinuousDistribution:
    assert n > 0 and m > 0, "n and m must be positive"

    def pdf(x):
        return (n/m)**(n/2) * x**(n/2-1) * (1+x*n/m)**(-(n+m)/2) / beta(n/2, m/2)

    def cdf(x):
        return betainc(n/2, m/2, (n*x)/(n*x+m))

    return ContinuousDistribution(pdf, cdf, f'F({n}, {m})')