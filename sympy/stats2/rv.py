from sympy import *

class PSpace(Basic):
    """
    Represents a mathematical probability space.

    We assume this probability space is complete.

    The event space is implied.
    For example, for a finite sample space,
    the event space is the powerset of the sample space.
    For a continuous sample space, the event space
    is the borel set of the sample space.

    Sample space is the non-empty set of all possible events.

    The probability measure assigns each event in the event space
    a number between 0 and 1. With the measure of the sample space
    equal to 1. Probability of the empty set is equal to 0.
    """
    is_Finite = None  # type: bool
    is_Continuous = None  # type: bool
    is_Discrete = None  # type: bool
    is_real = None  # type: bool

    def __new__(cls, sample_space: Set, probability_measure):

        return Basic.__new__(sample_space, probability_measure)

    sample_space = property(lambda self: self.args[0])
    probability_measure = property(lambda self: self.args[1])


class DiscretePSpace(PSpace):
    is_Finite = True


class RandomSymbol(AtomicExpr):
    def __new__(cls, symbol, pspace=None):
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        if not isinstance(symbol, Symbol):
            raise TypeError("symbol should be of type Symbol or string")
        if not isinstance(pspace, PSpace):
            raise TypeError("pspace variable should be of type PSpace")
        return Basic.__new__(cls, symbol, pspace)

    is_finite = True
    is_symbol = True

    _diff_wrt = True

    pspace = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])
    name   = property(lambda self: self.symbol.name)

    @property
    def is_commutative(self):
        return self.symbol.is_commutative

    @property
    def free_symbols(self):
        return {self}

