"""
Goals:

Unify crv, frv and drv subclasses.
Give pdf and pmf symbolic classes.
Remove all instances of `.args[i]` to improve readability.
Allow vector, matrix and stochastic distributions to be implemented more naturally.
Only compute integrals inside `PSpace` or `RandomDomain` but not both.
Make `PSpace` and its subclasses more consistent with itself and with its mathematical definition.
Keep all internal representations without a Piecewise function but add them for the user interface.
Add type hints.
Reduce pressure on the integrate module where practical (such as the Piecewise representations).

Computation for expectations happen in so many places and each one of them does their own checks.
These checks should be unified.
"""

__all__ = []

