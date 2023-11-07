# Statistics

This provides some useful tools to calculate cdf, quantiles, and scores of selected
distributions.
A user may also define one's own continuous distribution.

## Selected Distributions
The selected distributions are:
* [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
    via `normal_distribution(mu, sigma_squared)`

* [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) (with shape and rate)
    via `gamma_distribution(alpha, beta)`

* [Chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
    via `chi_squared_distribution(k)`

* [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
    via `student_t_distribution(nu)`

* [F-distribution](https://en.wikipedia.org/wiki/F-distribution)
    via `f_distribution(n, m)`


## Examples

```python
from distributions import *

Z = normal_distribution(0, 1)
# print f(0)
print(Z.get_pdf_at(0))
# print P(Z < 1)
print(Z.get_cdf_at(1))
# print P(0 <= Z <= 1)
print(Z.get_prob_between(0, 1))
# print the value z_{.05}
print(Z.get_score(.05))
# print Q3 for Z
print(Z.get_quantile(.75))

# print the value t_{.05,17}
print(student_t_distribution(17).get_score(.05))

# print the value F_{.01,5,6}
print(f_distribution(5, 6).get_score(.01))

# print the value F_{.01,5,6} with initial guess 4
print(gamma_distribution(7, 2).get_score(.01, guess=4))
```

## Defining Custom Distribution

To create a `ContinuousDistribution` object, you need to provide the _probability density function_ and the _cumulative distribution function_.
If a probability density function is $f(x)$, then the cumulative distribution function
$F(x)$ must satisfy $$F(x) = \int_{-\infty}^x f(t) \,\mathrm{d}t.$$

For instance, the exponential distribution with mean $1/\lambda$ has
$$f(x) = [x \ge 0]\lambda e^{-\lambda x}$$
and
$$F(x) = [x \ge 0](1-e^{-\lambda x}).$$

Hence, one may define as following.
```python
import math
from distributions import ContinuousDistribution

l = 1.5
my_distribution = ContinuousDistribution(
    pdf = lambda x: l * math.exp(-l * x) if x >= 0 else 0,
    cdf = lambda x: 1 - math.exp(-l * x) if x >= 0 else 0,
    name = f'Exp({1/l:.4g})'
)
```
Then, they could utilize some methods like `my_distribution.get_score(.05)`.
