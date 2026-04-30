from __future__ import annotations

from ..core.expression import (
    Add,
    Constant,
    Cos,
    Div,
    Exp,
    Expr,
    Ln,
    Mul,
    Neg,
    Pow,
    Sin,
    Sqrt,
    Sub,
    Symbol,
    Tan,
)
from ..core.simplify import simplify
from .differentiate import differentiate


def integrate(expr: Expr, var: str) -> Expr:
    """Compute the indefinite integral (antiderivative) of expr with respect to var.

    Uses a rule-based approach with the following techniques:
    - Power rule: ``int(x^n) = x^(n+1)/(n+1)``
    - Trigonometric: ``int(sin(x)) = -cos(x)``, ``int(cos(x)) = sin(x)``
    - Exponential: ``int(exp(x)) = exp(x)``
    - Logarithmic: ``int(1/x) = ln(x)``
    - Linearity: ``int(a*f + b*g) = a*int(f) + b*int(g)``
    - Integration by parts: ``int(f*g) = f*int(g) - int(f'*int(g))``

    The result is automatically simplified.

    Args:
        expr: The expression to integrate.
        var: The variable name to integrate with respect to.

    Returns:
        The simplified antiderivative (without the constant of integration).

    Raises:
        NotImplementedError: If no antiderivative can be found with the
            available rules.

    Examples:
        >>> from calcus import parse, integrate, pretty
        >>> pretty(integrate(parse("x^2"), "x"))
        'x ^ 3 / 3'
        >>> pretty(integrate(parse("sin(x)"), "x"))
        '-cos(x)'
        >>> pretty(integrate(parse("1/x"), "x"))
        'ln(x)'
    """
    result = _integrate_inner(expr, var)
    result = simplify(result)
    return result


def definite_integral(expr: Expr, var: str, lower: float, upper: float) -> float:
    """Compute the definite integral of expr from lower to upper bound.

    Attempts symbolic integration first using the Fundamental Theorem of
    Calculus (evaluate antiderivative at bounds). Falls back to numerical
    integration (Simpson's rule) if the symbolic approach fails.

    Args:
        expr: The expression to integrate.
        var: The variable name.
        lower: Lower bound of integration.
        upper: Upper bound of integration.

    Returns:
        The numerical value of the definite integral.

    Examples:
        >>> from calcus import parse, definite_integral
        >>> abs(definite_integral(parse("x^2"), "x", 0, 1) - 1/3) < 1e-10
        True
        >>> abs(definite_integral(parse("sin(x)"), "x", 0, 3.14159265) - 2) < 1e-8
        True
    """
    try:
        antideriv = integrate(expr, var)
        upper_val = antideriv.substitute(var, Constant(upper)).evaluate({})
        lower_val = antideriv.substitute(var, Constant(lower)).evaluate({})
        return upper_val - lower_val
    except NotImplementedError:
        from ..numerical.integrate import simpsons_rule

        return simpsons_rule(expr, var, lower, upper)


def _integrate_inner(expr: Expr, var: str) -> Expr:
    """Main integration dispatcher."""
    if isinstance(expr, Constant):
        return Mul.make(expr, Symbol(var))

    if isinstance(expr, Symbol):
        if expr.name == var:
            return Mul.make(Constant(0.5), Pow.make(expr, Constant(2)))
        return Mul.make(expr, Symbol(var))

    if isinstance(expr, Add):
        return Add.make(*(_integrate_inner(op, var) for op in expr.operands))

    if isinstance(expr, Mul):
        result = _integrate_mul(expr, var)
        if result is not None:
            return result

    if isinstance(expr, Neg):
        return Neg.make(_integrate_inner(expr.operand, var))

    if isinstance(expr, Pow):
        result = _integrate_pow(expr, var)
        if result is not None:
            return result

    if isinstance(expr, Sin):
        result = _integrate_sin(expr, var)
        if result is not None:
            return result

    if isinstance(expr, Cos):
        result = _integrate_cos(expr, var)
        if result is not None:
            return result

    if isinstance(expr, Exp):
        result = _integrate_exp(expr, var)
        if result is not None:
            return result

    if isinstance(expr, Div):
        result = _integrate_div(expr, var)
        if result is not None:
            return result

    raise NotImplementedError(
        f"Cannot integrate {type(expr).__name__}({expr}) with respect to {var}"
    )


def _integrate_mul(expr: Mul, var: str) -> Expr | None:
    """Try to integrate a product using constant pull-out and power rule."""
    operands = expr.operands
    symbols_in_var = [op for op in operands if op.symbols() and var in op.symbols()]
    constants = [op for op in operands if isinstance(op, Constant)]

    if len(symbols_in_var) == 1:
        inner = symbols_in_var[0]
        const_factor = Constant(1)
        for c in constants:
            const_factor = Mul.make(const_factor, c)

        result = _integrate_inner(inner, var)
        return Mul.make(const_factor, result)

    if len(symbols_in_var) == 0:
        result = Mul.make(*operands)
        return Mul.make(result, Symbol(var))

    if len(symbols_in_var) == 2:
        result = _integration_by_parts(expr, var)
        if result is not None:
            return result

    return None


def _integration_by_parts(expr: Mul, var: str) -> Expr | None:
    """Try integration by parts: int(f*g) = f*int(g) - int(f'*int(g))"""
    operands = expr.operands

    if len(operands) != 2:
        return None

    f, g = operands

    try:
        int_g = _integrate_inner(g, var)
        f_prime = differentiate(f, var)

        term1 = Mul.make(f, int_g)
        term2 = _integrate_inner(Mul.make(f_prime, int_g), var)

        return Sub.make(term1, term2)
    except NotImplementedError:
        pass

    try:
        int_f = _integrate_inner(f, var)
        g_prime = differentiate(g, var)

        term1 = Mul.make(g, int_f)
        term2 = _integrate_inner(Mul.make(g_prime, int_f), var)

        return Sub.make(term1, term2)
    except NotImplementedError:
        pass

    return None


def _integrate_pow(expr: Pow, var: str) -> Expr | None:
    """Integrate power expressions: int(x^n) = x^(n+1)/(n+1)."""
    base = expr.base
    exponent = expr.exponent

    if isinstance(base, Symbol) and base.name == var:
        if isinstance(exponent, Constant) and exponent.value != -1:
            new_exp = Constant(exponent.value + 1)
            return Div.make(Pow.make(base, new_exp), new_exp)
        if isinstance(exponent, Constant) and exponent.value == -1:
            return Ln.make(base)
        return None

    if isinstance(base, Constant) and isinstance(exponent, Symbol) and exponent.name == var:
        return Div.make(expr, Ln.make(base))

    return None


def _integrate_sin(expr: Sin, var: str) -> Expr | None:
    """Integrate sin expressions."""
    arg = expr.arg

    if isinstance(arg, Symbol) and arg.name == var:
        return Neg.make(Cos.make(arg))

    if isinstance(arg, Mul):
        result = _integrate_composite_trig(expr, var, Sin, Neg.make(Cos.make(arg)))
        if result is not None:
            return result

    return None


def _integrate_cos(expr: Cos, var: str) -> Expr | None:
    """Integrate cos expressions."""
    arg = expr.arg

    if isinstance(arg, Symbol) and arg.name == var:
        return Sin.make(arg)

    if isinstance(arg, Mul):
        result = _integrate_composite_trig(expr, var, Cos, Sin.make(arg))
        if result is not None:
            return result

    return None


def _integrate_composite_trig(
    expr: Sin | Cos,
    var: str,
    trig_type: type,
    antideriv: Expr,
) -> Expr | None:
    """Handle int(trig(k*x)) = antideriv/k for constant k."""
    arg = expr.arg

    if not isinstance(arg, Mul):
        return None

    operands = arg.operands
    symbols = [op for op in operands if isinstance(op, Symbol) and op.name == var]
    constants = [op for op in operands if isinstance(op, Constant)]

    if len(symbols) == 1 and len(constants) >= 1:
        k = Constant(1)
        for c in constants:
            k = Mul.make(k, c)

        if isinstance(k, Constant):
            return Div.make(antideriv, k)

    return None


def _integrate_exp(expr: Exp, var: str) -> Expr | None:
    """Integrate exp expressions."""
    arg = expr.arg

    if isinstance(arg, Symbol) and arg.name == var:
        return Exp.make(arg)

    if isinstance(arg, Mul):
        operands = arg.operands
        symbols = [op for op in operands if isinstance(op, Symbol) and op.name == var]
        constants = [op for op in operands if isinstance(op, Constant)]

        if len(symbols) == 1 and len(constants) >= 1:
            k = Constant(1)
            for c in constants:
                k = Mul.make(k, c)

            if isinstance(k, Constant):
                return Div.make(Exp.make(arg), k)

    return None


def _integrate_div(expr: Div, var: str) -> Expr | None:
    """Try to integrate division expressions."""
    num = expr.numerator
    den = expr.denominator

    if isinstance(den, Symbol) and den.name == var:
        if isinstance(num, Constant):
            return Mul.make(num, Ln.make(den))

    if isinstance(den, Pow) and isinstance(den.base, Symbol) and den.base.name == var:
        if isinstance(den.exponent, Constant):
            if isinstance(num, Constant):
                if den.exponent.value == 1:
                    return Mul.make(num, Ln.make(den.base))
                new_exp = Constant(-den.exponent.value + 1)
                return Mul.make(num, Div.make(Pow.make(den.base, new_exp), new_exp))

    return None
