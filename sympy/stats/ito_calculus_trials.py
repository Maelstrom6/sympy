from __future__ import print_function, division

from functools import singledispatch

from sympy import (sympify, Matrix, MatrixSymbol, S, Indexed, Basic,
                   Set, And, Eq, FiniteSet, ImmutableMatrix, symbols,
                   Lambda, Mul, Dummy, IndexedBase, Add, Interval, oo,
                   linsolve, eye, Or, Not, Intersection, factorial, Contains,
                   Union, Expr, Function, exp, cacheit, sqrt, pi, gamma,
                   Ge, Piecewise, Symbol, NonSquareMatrixError, EmptySet,
                   Subs, Array, Integer, ordered, MatrixBase, NDimArray, MatrixExpr,
                   Derivative, diff, integrate, Integral)
from sympy.core.compatibility import reduce
from sympy.stats.rv import is_random, RandomIndexedSymbol
from sympy.stats.stochastic_process_types import WienerProcess
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.basic import _atomic
from sympy.core.containers import Tuple
from sympy.utilities.misc import filldedent
from collections import Counter
from sympy.calculus.util import continuous_domain


@is_random.register(Integral)
def _(x):
    f = is_random(x.function)
    s = any([is_random(v) for v in x.limits[0]])
    return Or(f, s)


class Differential(Expr):
    is_Differential = True

    def __new__(cls, expr, *variables, **kwargs):
        # from sympy.stats import WienerProcess
        expr = sympify(expr)
        symbols_or_none = getattr(expr, "free_symbols", None)
        has_symbol_set = isinstance(symbols_or_none, set)

        if not has_symbol_set:
            raise ValueError(filldedent('''
                Since there are no variables in the expression %s,
                the differential cannot be found.''' % expr))

        # determine value for variables if it wasn't given
        if not variables:
            variables = expr.free_symbols
            if len(variables) != 1:
                if expr.is_number:
                    return S.Zero
                if len(variables) == 0:
                    raise ValueError(filldedent('''
                        Since there are no variables in the expression,
                        the variable(s) of the differential must be supplied
                        for %s''' % expr))
                else:
                    raise ValueError(filldedent('''
                        Since there is more than one variable in the
                        expression, the variable(s) of the differential
                        must be supplied for %s''' % expr))

        # Standardize the variables by sympifying them:
        variables = list(sympify(variables))

        # Split the list of variables into a list of the variables we are diff
        # wrt, where each element of the list has the form (s, count) where
        # s is the entity to diff wrt and count is the order of the
        # differential.
        variable_count = []
        array_likes = (tuple, list, Tuple)

        for i, v in enumerate(variables):
            if isinstance(v, Integer):
                if i == 0:
                    raise ValueError("First variable cannot be a number: %i" % v)
                count = v
                prev, prevcount = variable_count[-1]
                if prevcount != 1:
                    raise TypeError("tuple {} followed by number {}".format((prev, prevcount), v))
                if count == 0:
                    variable_count.pop()
                else:
                    variable_count[-1] = Tuple(prev, count)
            else:
                if isinstance(v, array_likes):
                    if len(v) == 0:
                        # Ignore empty tuples: Differential(expr, ... , (), ... )
                        continue
                    if isinstance(v[0], array_likes):
                        # Derive by array: Differential(expr, ... , [[x, y, z]], ... )
                        if len(v) == 1:
                            v = Array(v[0])
                            count = 1
                        else:
                            v, count = v
                            v = Array(v)
                    else:
                        v, count = v
                    if count == 0:
                        continue
                elif isinstance(v, UndefinedFunction):
                    raise TypeError(
                        "cannot find the differential wrt "
                        "UndefinedFunction: %s" % v)
                else:
                    count = 1
                variable_count.append(Tuple(v, count))

        # light evaluation of contiguous, identical
        # items: (x, 1), (x, 1) -> (x, 2)
        merged = []
        for t in variable_count:
            v, c = t
            if c.is_negative:
                raise ValueError(
                    'order of differentiation must be nonnegative')
            if merged and merged[-1][0] == v:
                c += merged[-1][1]
                if not c:
                    merged.pop()
                else:
                    merged[-1] = Tuple(v, c)
            else:
                merged.append(t)
        variable_count = merged

        # sanity check of variables of differentation; we waited
        # until the counts were computed since some variables may
        # have been removed because the count was 0
        for v, c in variable_count:
            # v must have _diff_wrt True
            if not v._diff_wrt:
                __ = ''  # filler to make error message neater
                raise ValueError(filldedent('''
                    Can't calculate differential wrt %s.%s''' % (v,
                                                                 __)))

        # We make a special case for 0th differential, because there is no
        # good way to unambiguously print this.
        if len(variable_count) == 0:
            return expr

        evaluate = kwargs.get('evaluate', False)

        if evaluate:
            if isinstance(expr, Differential):
                expr = expr.canonical
            variable_count = [
                (v.canonical if isinstance(v, Differential) else v, c)
                for v, c in variable_count]
            # Look for a quick exit if there are symbols that don't appear in
            # expression at all. Note, this cannot check non-symbols like
            # Differentials as those can be created by intermediate
            # differentials.
            zero = False
            free = expr.free_symbols
            for v, c in variable_count:
                vfree = v.free_symbols
                if c.is_positive and vfree:
                    if isinstance(v, AppliedUndef):
                        # these match exactly since
                        # x.diff(f(x)) == g(x).diff(f(x)) == 0
                        # and are not created by differentiation
                        dum = Dummy()
                        if not expr.xreplace({v: dum}).has(dum):
                            zero = True
                            break
                    elif isinstance(v, MatrixExpr):
                        zero = False
                        break
                    elif isinstance(v, Symbol) and v not in free:
                        zero = True
                        break
                    else:
                        if not free & vfree:
                            # e.g. v is IndexedBase or Matrix
                            zero = True
                            break
            if zero:
                if isinstance(expr, (MatrixBase, NDimArray)):
                    return expr.zeros(*expr.shape)
                elif isinstance(expr, MatrixExpr):
                    from sympy import ZeroMatrix
                    return ZeroMatrix(*expr.shape)
                elif expr.is_scalar:
                    return S.Zero

            # make the order of symbols canonical
            variable_count = cls._sort_variable_count(variable_count)

        # denest
        if isinstance(expr, Differential):
            variable_count = list(expr.variable_count) + variable_count
            expr = expr.expr
            return Differential(expr, *variable_count, **kwargs)

        # we return here if evaluate is False or if there is no
        # _eval_differential method
        if not evaluate or not hasattr(expr, '_eval_differential'):
            # return an unevaluated Differential
            return Expr.__new__(cls, expr, *variable_count)

        # what we have so far can be made canonical
        expr = expr.replace(
            lambda x: isinstance(x, Differential),
            lambda x: x.canonical)
        return Expr.__new__(cls, expr, *variable_count)

    @property
    def canonical(self):
        return self.func(self.expr,
                         *Differential._sort_variable_count(self.variable_count))

    @classmethod
    def _sort_variable_count(cls, vc):
        from sympy.utilities.iterables import uniq, topological_sort
        if not vc:
            return []
        vc = list(vc)
        if len(vc) == 1:
            return [Tuple(*vc[0])]
        V = list(range(len(vc)))
        E = []
        v = lambda i: vc[i][0]
        D = Dummy()

        def _block(d, v, wrt=False):
            # return True if v should not come before d else False
            if d == v:
                return wrt
            if d.is_Symbol:
                return False
            if isinstance(d, Differential):
                # a differential blocks if any of it's variables contain
                # v; the wrt flag will return True for an exact match
                # and will cause an AppliedUndef to block if v is in
                # the arguments
                if any(_block(k, v, wrt=True)
                       for k in d._wrt_variables):
                    return True
                return False
            if not wrt and isinstance(d, AppliedUndef):
                return False
            if v.is_Symbol:
                return v in d.free_symbols
            if isinstance(v, AppliedUndef):
                return _block(d.xreplace({v: D}), D)
            return d.free_symbols & v.free_symbols

        for i in range(len(vc)):
            for j in range(i):
                if _block(v(j), v(i)):
                    E.append((j, i))
        # this is the default ordering to use in case of ties
        O = dict(zip(ordered(uniq([i for i, c in vc])), range(len(vc))))
        ix = topological_sort((V, E), key=lambda i: O[v(i)])
        # merge counts of contiguously identical items
        merged = []
        for v, c in [vc[i] for i in ix]:
            if merged and merged[-1][0] == v:
                merged[-1][1] += c
            else:
                merged.append([v, c])
        return [Tuple(*i) for i in merged]

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    # no _eval_mul()?
    def __mul__(self, other):
        if isinstance(other, self.func):
            if self != other:
                return 0
        if isinstance(other, Differential):
            if isinstance(self.expr, RandomIndexedSymbol) and isinstance(other.expr,
                                                                         RandomIndexedSymbol):
                if isinstance(self.expr.pspace.process, WienerProcess) and isinstance(
                    other.expr.pspace.process, WienerProcess):
                    return Piecewise((d(self.expr.args[0].args[0], self.variable_count),
                                      Eq(self.expr.args[0].args[0], other.expr.args[0].args[0]),
                                      (0, True)))
        return Mul(self, other)

    def _eval_power(self, other):
        if other > 2:
            return 0
        if other == 2 and isinstance(self.expr, RandomIndexedSymbol):
            if isinstance(self.expr.pspace.process, WienerProcess):
                return d(self.expr.args[0].args[0], self.variable_count)
        else:
            return 0

    def doit(self, **hints):
        expr = self.expr
        if hints.get('deep', True):
            expr = expr.doit(**hints)
        hints['evaluate'] = True
        rv = self.func(expr, *self.variable_count, **hints)
        if rv != self and rv.has(Differential):
            rv = rv.doit(**hints)
        return rv

    @property
    def expr(self):
        return self._args[0]

    @property
    def _wrt_variables(self):
        # return the variables of differentiation without
        # respect to the type of count (int or symbolic)
        return [i[0] for i in self.variable_count]

    @property
    def variables(self):
        # TODO: deprecate?  YES, make this 'enumerated_variables' and
        #       name _wrt_variables as variables
        # TODO: support for `d^n`?
        rv = []
        for v, count in self.variable_count:
            if not count.is_Integer:
                raise TypeError(filldedent('''
                    Cannot give expansion for symbolic count. If you just
                    want a list of all variables of differentiation, use
                    _wrt_variables.'''))
            rv.extend([v] * count)
        return tuple(rv)

    @property
    def variable_count(self):
        return self._args[1:]

    @property
    def differential_count(self):
        return sum([count for var, count in self.variable_count], 0)

    @property
    def free_symbols(self):
        ret = self.expr.free_symbols
        # Add symbolic counts to free_symbols
        for var, count in self.variable_count:
            ret.update(count.free_symbols)
        return ret

    def _eval_subs(self, old, new):
        # The substitution (old, new) cannot be done inside
        # Differential(expr, vars) for a variety of reasons
        # as handled below.
        if old in self._wrt_variables:
            # first handle the counts
            expr = self.func(self.expr, *[(v, c.subs(old, new))
                                          for v, c in self.variable_count])
            if expr != self:
                return expr._eval_subs(old, new)
            # quick exit case
            if not getattr(new, '_diff_wrt', False):
                # case (0): new is not a valid variable of
                # differentiation
                if isinstance(old, Symbol):
                    # don't introduce a new symbol if the old will do
                    return Subs(self, old, new)
                else:
                    xi = Dummy('xi')
                    return Subs(self.xreplace({old: xi}), xi, new)

        # If both are Differentials with the same expr, check if old is
        # equivalent to self or if old is a subdifferential of self.
        if old.is_Differential and old.expr == self.expr:
            if self.canonical == old.canonical:
                return new

            # collections.Counter doesn't have __le__
            def _subset(a, b):
                return all((a[i] <= b[i]) == True for i in a)

            old_vars = Counter(dict(reversed(old.variable_count)))
            self_vars = Counter(dict(reversed(self.variable_count)))
            if _subset(old_vars, self_vars):
                return Differential(new, *(self_vars - old_vars).items()).canonical

        args = list(self.args)
        newargs = list(x._subs(old, new) for x in args)
        if args[0] == old:
            # complete replacement of self.expr
            # we already checked that the new is valid so we know
            # it won't be a problem should it appear in variables
            return Differential(*newargs)

        if newargs[0] != args[0]:
            # case (1) can't change expr by introducing something that is in
            # the _wrt_variables if it was already in the expr
            # e.g.
            # for Differential(f(x, g(y)), y), x cannot be replaced with
            # anything that has y in it; for f(g(x), g(y)).diff(g(y))
            # g(x) cannot be replaced with anything that has g(y)
            syms = {vi: Dummy() for vi in self._wrt_variables
                    if not vi.is_Symbol}
            wrt = {syms.get(vi, vi) for vi in self._wrt_variables}
            forbidden = args[0].xreplace(syms).free_symbols & wrt
            nfree = new.xreplace(syms).free_symbols
            ofree = old.xreplace(syms).free_symbols
            if (nfree - ofree) & forbidden:
                return Subs(self, old, new)

        viter = ((i, j) for ((i, _), (j, _)) in zip(newargs[1:], args[1:]))
        if any(i != j for i, j in viter):  # a wrt-variable change
            # case (2) can't change vars by introducing a variable
            # that is contained in expr, e.g.
            # for Differential(f(z, g(h(x), y)), y), y cannot be changed to
            # x, h(x), or g(h(x), y)
            for a in _atomic(self.expr, recursive=True):
                for i in range(1, len(newargs)):
                    vi, _ = newargs[i]
                    if a == vi and vi != args[i][0]:
                        return Subs(self, old, new)
            # more arg-wise checks
            vc = newargs[1:]
            oldv = self._wrt_variables
            newe = self.expr
            subs = []
            for i, (vi, ci) in enumerate(vc):
                if not vi._diff_wrt:
                    # case (3) invalid differentiation expression so
                    # create a replacement dummy
                    xi = Dummy('xi_%i' % i)
                    # replace the old valid variable with the dummy
                    # in the expression
                    newe = newe.xreplace({oldv[i]: xi})
                    # and replace the bad variable with the dummy
                    vc[i] = (xi, ci)
                    # and record the dummy with the new (invalid)
                    # differentiation expression
                    subs.append((xi, vi))

            if subs:
                # handle any residual substitution in the expression
                newe = newe._subs(old, new)
                # return the Subs-wrapped differential
                return Subs(Differential(newe, *vc), *zip(*subs))

        # everything was ok
        return Differential(*newargs)

    @property
    def _form_field(self):
        # hack for the printer that assumes
        # Differential comes from the geom module.
        return self.expr


def d(f, *symbols, **kwargs):
    """

    Parameters
    ==========
    f
    symbols
    kwargs

    Returns
    =======

    Examples
    ========

    >>> from sympy.stats import WienerProcess
    >>> from sympy import symbols
    >>> a, t = symbols('a t', positive=True)
    >>> W = WienerProcess('W')
    >>> dWt = d(W(t+a) + 2, t)
    d(W(t+a), t)
    >>> dt = d(t + 1)
    >>> dWt * dt
    0
    >>> dWt ** 2
    d(t)
    >>> dt ** 2
    0


    """
    if hasattr(f, 'differential'):
        return f.differential(*symbols, **kwargs)
    if isinstance(f, Add):
        return Add(*[d(a, *symbols, **kwargs) for a in f.args])

    if isinstance(f, Mul):
        args = list(f.args)
        terms = []
        for i in range(len(args)):
            result = d(args[i], *symbols, **kwargs)
            if result:
                # Note: reduce is used in step of Mul as Mul is unable to
                # handle subtypes and operation priority:
                terms.append(
                    reduce(lambda x, y: x * y, (args[:i] + [result] + args[i + 1:]), S.One))
        return Add.fromiter(terms)

    kwargs.setdefault('evaluate', True)
    return Differential(f, *symbols, **kwargs)


def is_ito_function(f: Expr, Wt=None, t=None):
    if isinstance(f, Function):
        return True

    x = symbols('x', real=True)
    if (Wt is None) and (t is None):
        Wts = [sym for sym in f.free_symbols if isinstance(sym, RandomIndexedSymbol)]
        Wts = [sym for sym in Wts if isinstance(sym.pspace.process, WienerProcess)]
        if len(Wts) != 1:
            raise ValueError("Must have one Wt")
        t = (f.free_symbols - set(Wts)).pop()
        Wt = Wts[0]

    f = f.replace(Wt, x)

    f_x = f.diff(x)
    f_xx = f_x.diff(x)
    f_t = f.diff(t)
    f_x_continuous = continuous_domain(f_x, x, S.Reals).intersect(S.Reals) != EmptySet
    f_xx_continuous = continuous_domain(f_xx, x, S.Reals).intersect(S.Reals) != EmptySet
    f_t_continuous = continuous_domain(f_t, t, S.Reals).intersect(S.Reals) != EmptySet

    FWB_measurable = True
    FW_adapted = True
    T = symbols("T", real=True, positive=True)
    square_integrable = (integrate(f ** 2, (t, 0, T)) < oo is not False)  # can be None

    return And(f_x_continuous, f_xx_continuous, f_t_continuous,
               FWB_measurable, FW_adapted, square_integrable)


class ItoIntegral(Integral):
    __slots__ = ('is_commutative',)

    def __new__(cls, function, symbol, **assumptions):
        if hasattr(function, '_eval_ItoIntegral'):
            return function._eval_ItoIntegral(*symbols, **assumptions)

        syms = [sym for sym in symbol[0].expr_free_symbols if isinstance(sym, Symbol)]
        if len(syms) != 1:
            raise ValueError("The integrator must be a monotonic function of "
                             "exactly one variable.")

        obj = Expr.__new__(cls, function, symbol, **assumptions)
        return obj

    @property
    def is_random(self):
        return is_random(self)

    @property
    def alpha(self):
        return self.variables[0]

    def doit(self, **hints):
        if not Or(is_random(self.function), is_random(self.alpha)):
            # This is then a Riemann-Stieltjes integral.
            # Assume alpha is monotonic.
            x = self.alpha.free_symbols.pop()
            new_symbols = (x, self.limits[0][1], self.limits[0][2])
            return integrate(self.function * self.alpha.diff(x), new_symbols)
        raise NotImplementedError()


def ito_integrate(*args, **kwargs):
    doit_flags = {
        'deep': False,
        'meijerg': kwargs.pop('meijerg', None),
        'conds': kwargs.pop('conds', 'piecewise'),
        'risch': kwargs.pop('risch', None),
        'heurisch': kwargs.pop('heurisch', None),
        'manual': kwargs.pop('manual', None)
    }
    integral = ItoIntegral(*args, **kwargs)

    if isinstance(integral, ItoIntegral):
        return integral.doit(**doit_flags)
    else:
        new_args = [a.doit(**doit_flags) if isinstance(a, ItoIntegral) else a
                    for a in integral.args]
        return integral.func(*new_args)


from sympy import pdsolve, dsolve, solve


def ito_solve(eq, Xt=None, hint='default', **kwargs):
    # print(ito_solve(Eq(d(g, t), mu * d(t) + sigma * d(W(t), t))))
    # https://www.researchgate.net/publication/45267258_Algorithmic_Solution_of_Stochastic_Differential_Equations
    # with m=1 and d=1
    prep = kwargs.pop('prep', True)
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs

    # identify WienerProcess and t
    Wt_ris = eq.atoms(RandomIndexedSymbol).pop()
    t = Wt_ris.key
    Wt_sym, dWt_sym, dt_sym, Xt_sym = symbols('Wt dWt dt Xt', real=True)
    Wt_func = Function('W')(t)
    dWt = d(Wt_func)  # dWt
    dt = d(t)  # dt

    # remove W(t)
    eq = eq.replace(Wt_ris, Wt_func).replace(d(Wt_ris, t), dWt)
    if Xt is not None:
        func = Xt.replace(Wt_ris, Wt_func)

    # remove d(x(t)) for dx and others
    def to_symbols(e, f):
        e = e.replace(dWt, dWt_sym).replace(Wt_func, Wt_sym).replace(dt, dt_sym)
        if f is not None:
            f = f.replace(Wt_func, Wt_sym)
        else:
            f = None
        return e, f

    def to_functions(e, f):
        e = e.replace(Wt_sym, Wt_func).replace(dWt_sym, dWt).replace(dt_sym, dt)
        if f is not None:
            f = f.replace(Wt_sym, Wt_func)
        else:
            f = None
        return e, f

    eq, Xt = to_symbols(eq, Xt)

    drift = eq.coeff(dt_sym, 1)  # f(t, X(t)) from references
    diffusion = eq.coeff(dWt_sym, 1)  # g1(t, X(t)) from references

    # adjusted from _preprocess
    differentials = eq.atoms(Differential)
    if not Xt:
        funcs = set().union(*[di.atoms(AppliedUndef) for di in differentials])
        if len(funcs) != 1:
            raise ValueError('The function cannot be '
                             'automatically detected for %s.' % eq)
        Xt = funcs.pop()  # X(t)

    drift = drift.replace(Xt, Xt_sym)
    diffusion = diffusion.replace(Xt, Xt_sym)
    print("drift", drift)
    print("diffusion", diffusion)

    # if Xt is not a function of t but only Wt
    # runs if d(Xt) is not written in terms of Xt
    if (diffusion.replace(Xt_sym, Xt).diff(Wt_sym) / 2 - drift.replace(Xt_sym, Xt)).simplify() == 0:
        return ito_solve_timeless(Xt, Wt_ris, Wt_sym, drift, diffusion)

    Xt_func = Function('Xt')
    alpha_t = Function('alpha', real=True)(t)
    beta_Wt = Function('beta', real=True)(Wt_sym)
    alpha_0, beta_0 = symbols('alpha_0 beta_0', real=True)
    # alpha_t, beta_Wt = symbols('alpha_t beta_Wt', real=True)
    X0 = symbols('X_0', real=True)

    # Xt_alpha = dsolve(Xt_func(Wt_sym).diff(Wt_sym) + diffusion.replace(Xt_sym, Xt_func(Wt_sym)),
    #                   Xt_func(Wt_sym), ics={Xt_func(0): alpha_0}).rhs.subs(alpha_0, alpha_t)
    Xt_alpha = dsolve(Xt_func(Wt_sym).diff(Wt_sym) + diffusion.replace(Xt_sym, Xt_func(Wt_sym)),
                      Xt_func(Wt_sym)).rhs.subs('C1', alpha_t)

    print(solve(Eq(Xt_alpha.subs(alpha_t, alpha_0).subs(Wt_sym, 0).subs(t, 0), X0), alpha_0))
    print(Xt_alpha.diff(Wt_sym))

    half_ddx_squared = Xt_alpha.diff(Wt_sym, 2) / 2
    # Xt_beta = dsolve(Xt_func(t).diff(t) + half_ddx_squared + drift.replace(Xt_sym, Xt_func(t)),
    #        Xt_func(t), ics={Xt_func(0): beta_0}).rhs.subs(beta_0, beta_Wt)
    Xt_beta = dsolve(Xt_func(t).diff(t) + half_ddx_squared + drift.replace(Xt_sym, Xt_func(t)),
                     Xt_func(t)).rhs.subs('C1', beta_Wt)

    # we need to now find alpha(t) and beta(Wt) by noting that Xt_alpha == Xt_beta

    print(Xt_alpha.simplify())
    print(Xt_beta.simplify())

    alpha_t_star = dsolve(Eq(Xt_alpha.diff(t), Xt_beta.diff(t)),
                          alpha_t).rhs.subs('C1', X0)

    print(alpha_t_star)

    solution = Xt_alpha.subs(alpha_t, alpha_t_star)
    return Eq(Xt, solution).replace(Wt_sym, Wt_ris)


def ito_solve_timeless(Xt, Wt_ris, Wt_sym, drift, diffusion):
    F = Function('F', real=True)  # not a function of t
    X0 = symbols('X0', real=True)
    solution = dsolve(F(Wt_sym).diff(Wt_sym) + diffusion, F(Wt_sym), ics={F(0): X0}).rhs
    return Eq(Xt, solution).replace(Wt_sym, Wt_ris)


def itos_lemma(Xt):
    # eg, exp(W(t))
    # must be a function of W(t) and t
    # identify WienerProcess and t
    Wt_ris = Xt.atoms(RandomIndexedSymbol).pop()
    t = Wt_ris.key
    Wt_sym, dWt_sym, dt_sym, Xt_sym = symbols('Wt dWt dt Xt', real=True)
    Wt_func = Function('W')(t)
    dWt = d(Wt_func)  # dWt
    dt = d(t)  # dt

    Xt_expr = Xt.replace(Wt_ris, Wt_sym)
    drift = Xt_expr.diff(t) + Xt_expr.diff(Wt_sym, 2) / 2
    diffusion = Xt_expr.diff(Wt_sym)
    return Eq(d(Xt, t), drift * d(t) + diffusion * d(Wt_ris, t))


a, t = symbols('a t', positive=True)
W = WienerProcess('W')
g = Function('g')(W(t), t)
mu, sigma = symbols('mu sigma', real=True)

# print(ito_solve(Eq(d(g, t), g*d(t) + g*d(W(t), t)/2)))
print(ito_solve(Eq(d(g, t), mu * d(t) + sigma * d(W(t), t))))
print(ito_solve(Eq(d(g, t), (0 + sigma ** 2 / 2) * g * d(t) + sigma * g * d(W(t), t))))
