from sympy import *
from sympy.logic.boolalg import Boolean


class PSpace(Basic):
    """
    One property that is implied is the event space, F which needs to be a sigma field.
    F is assumed to be the powerset of sample space if the
    sample space is finite and is assumed to be the Borel set of the sample space if infinite.

    This means that we assume all probability spaces are complete probability spaces. (I assume)

    probability_measure is a measure: F -> [0, 1] and
    is sometimes called the probability distribution in the discrete case.
    """
    def __new__(cls, sample_space: Set, probability_measure: Lambda):
        ...

    @property
    def sample_space(self) -> Set:
        return self.args[0]

    @property
    def probability_measure(self) -> Lambda:
        return self.args[1]


class RandomSymbol(AtomicExpr):
    """
    A random variable is a function: Omega -> S where Omega is a sample space and S is a state space.
    At SymPy we adopt a slightly different interpretation to make computation easier.
    We set the sample space equal to the state space and the random variable takes on the identity function.
    This makes inversion a lot easier as well.

    We will assume all random variables are F-measurable over its probability space.

    May add original_pspace and derived_pspace here.
    original_pspace will be used to check for independence between other random variables.
    derived_pspace will be used for computation by using many distributional identities. This value can be None.
    """
    def __new__(cls, symbol: Symbol, p_space: PSpace):
        ...

    @property
    def symbol(self) -> Symbol:
        return self.args[0]

    @property
    def p_space(self) -> PSpace:
        return self.args[1]

    def free_symbols(self):
        return {self.symbol}

    @property
    def f(self) -> Lambda:
        """
        The likelihood function of the random variable.
        - In the discrete case, this is the PMF and is
          f(x) = P({w in Omega : X(w) = x}) abbreviated to P(X = x).
        - In the continuous case, this is the PDF, that is, f(x) = F'(x).
        - For a mixed case, we will use the PDF together with DiracDelta's.

        This will be achieved by interpreting the probability measure of self.p_space
        These will not return piecewise functions since they are internal.
        Only the user interface will use piecewise functions.

        The domain is implied to be the sample space itself.
        It is 0 for all values not in the sample space.

        If X and x are vectors, then the equality in the PMF is taken pointwise and
        the derivative in the PDF is taken once with respect to each element in the vector.
        """
        ...

    @property
    def F(self) -> Lambda:
        """
        The distribution function of the random variable.
        That is, a function F(x) = P({w in Omega : X(w) <= x})
        which is abbreviated to P(X <= x).

        The domain is implied to be the interval from the minimum value of
        the sample space to the maximum value of the sample space.
        It is 0 for values less than the minimum value of the sample space
        and 1 for all values greater than the maximum value of the sample space.

        This will be achieved by interpreting the probability measure of self.p_space
        These will not return piecewise functions since they are internal.
        Only the user interface will use piecewise functions.

        Other properties:
        All random variables have distribution functions
        F(x) is weakly increasing
        F(x) is right continuous
        lim x->-oo F(x) = 0
        lim x->oo F(x) = 1

        If X and x are vectors, then the inequality is taken pointwise.
        """
        ...


class RandomRelational(Boolean):
    """
    Represents any value inside a probability statement
    which is also any value inside a given statement.

    All kinds of statements like this will be converted to RandomRelational.

    This includes things like X == x, X < Y, X <= x, X.
    """
    def as_boolean(self):
        """
        The same kind of function that appears in the old stats module.
        """
        ...


