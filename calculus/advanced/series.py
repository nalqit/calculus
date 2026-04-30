from __future__ import annotations

from ..core.expression import (
    Add,
    Constant,
    Div,
    Expr,
    Mul,
    Pow,
    Symbol,
)
from ..core.simplify import simplify
from ..symbolic.differentiate import differentiate


def taylor_series(
    expr: Expr, var: str, point: float = 0, order: int = 5
) -> Expr:
    """Compute the Taylor series expansion of expr around a point.

    The Taylor series of f(x) around x=a is:
    ``f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2! + f'''(a)(x-a)^3/3! + ...``

    Args:
        expr: The expression to expand.
        var: The variable name.
        point: The expansion point (default 0, giving Maclaurin series).
        order: Number of terms to include (default 5).

    Returns:
        A polynomial expression approximating the original function.
        Terms with zero coefficients are omitted.

    Raises:
        NotImplementedError: If a derivative cannot be computed.

    Examples:
        >>> from calculus import parse, taylor_series, pretty
        >>> pretty(taylor_series(parse("exp(x)"), "x", order=5))
        '1 + x + x ^ 2 / 2 + x ^ 3 / 6 + x ^ 4 / 24'
    """
    terms: list[Expr] = []
    current = expr
    factorial = 1

    for n in range(order):
        if n > 0:
            current = differentiate(current, var)
            factorial *= n

        coeff_val = _evaluate_at(current, var, point)
        coeff = Constant(coeff_val)

        if abs(coeff_val) > 1e-15:
            term = Mul.make(coeff, Pow.make(Symbol(var), Constant(n)))
            if n > 0:
                term = Div.make(term, Constant(factorial))
            terms.append(term)

    if not terms:
        return Constant(0)

    return simplify(Add.make(*terms))


def maclaurin_series(expr: Expr, var: str, order: int = 5) -> Expr:
    """Compute the Maclaurin series expansion (Taylor series at x=0).

    A convenience wrapper around ``taylor_series(expr, var, point=0)``.
    The Maclaurin series is commonly used for standard function expansions.

    Args:
        expr: The expression to expand.
        var: The variable name.
        order: Number of terms to include (default 5).

    Returns:
        A polynomial expression approximating the function near x=0.

    Examples:
        >>> from calculus import parse, maclaurin_series, pretty
        >>> pretty(maclaurin_series(parse("sin(x)"), "x", order=6))
        'x - x ^ 3 / 6 + x ^ 5 / 120'
    """
    return taylor_series(expr, var, point=0, order=order)


def _evaluate_at(expr: Expr, var: str, value: float) -> float:
    """Evaluate expr with var = value."""
    substituted = expr.substitute(var, Constant(value))
    return substituted.evaluate({})



