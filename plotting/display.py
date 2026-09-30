"""How the paper prints numbers.

Estimates and parameters are printed to a number of significant digits. Certified bounds are
rounded outward, a lower bound down and an upper bound up, so a printed lower bound never exceeds,
and a printed upper bound never falls below, the certified value.
"""
import math
from decimal import Decimal
from fractions import Fraction


def pct(v, digits=3):
    """`v` to `digits` significant digits; values of 100 or more without decimals."""
    return f'{v:.0f}' if abs(v) >= 100 else f'{v:.{digits}g}'


def sci(v):
    """`v` in LaTeX power-of-ten notation with two significant digits."""
    if v == 0:
        return '0'
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    if round(m, 1) >= 10:  # the mantissa rounds up to 10.0
        m, e = m / 10, e + 1
    return f'{m:.1f}\\times10^{{{e}}}'


def directed(v, digits=3, up=False, decimals=None):
    """`v` rounded outward to `digits` significant digits, or to `decimals` decimals: toward +inf if
    `up`, else toward -inf. An exact rational is rounded exactly; a float is first moved two ulps in
    the rounding direction, so the result stays on the conservative side of the certified value."""
    if isinstance(v, float):
        for _ in range(2):
            v = math.nextafter(v, math.inf if up else -math.inf)
    x = Fraction(v)
    if x == 0:
        return '0'
    if decimals is None:
        decimals = digits - 1 - math.floor(math.log10(abs(float(x))))
    scale = Fraction(10) ** decimals
    k = math.ceil(x * scale) if up else math.floor(x * scale)
    return format(Decimal(k).scaleb(-decimals), 'f')


def lo(v, digits=3, decimals=None):
    """A certified lower bound or lower endpoint, rounded down."""
    return directed(v, digits, False, decimals)


def hi(v, digits=3, decimals=None):
    """A certified upper bound or upper endpoint, rounded up."""
    return directed(v, digits, True, decimals)
