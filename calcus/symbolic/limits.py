from __future__ import annotations

from ..core.expression import (
    Add,
    Constant,
    Cos,
    Div,
    Expr,
    Function,
    Ln,
    Mul,
    Neg,
    Pow,
    Sin,
    Sub,
    Symbol,
)
from ..core.simplify import simplify
from .differentiate import differentiate


INFINITY = float("inf")
NEG_INFINITY = float("-inf")


def limit(
    expr: Expr, var: str, point: float | str, direction: str = "both"
) -> float:
    """Compute the limit of expr as var approaches point.

    Uses the following strategies in order:
    1. Direct substitution: Evaluate the expression at the limit point.
    2. L'Hôpital's rule: For 0/0 or ∞/∞ indeterminate forms.
    3. Numerical epsilon-delta approximation.

    Args:
        expr: The expression.
        var: The variable name.
        point: The limit point. A number for finite limits, or the strings
            ``"inf"`` or ``"-inf"`` for limits at infinity.
        direction: ``"both"`` (default), ``"left"``, or ``"right"``.

    Returns:
        The computed limit value as a float.

    Raises:
        ValueError: If the limit does not exist or cannot be computed.

    Examples:
        >>> from calcus import parse, limit
        >>> limit(parse("x^2 + 1"), "x", 3)
        10.0
        >>> abs(limit(parse("sin(x)/x"), "x", 0) - 1) < 1e-6
        True
        >>> limit(parse("1/x"), "x", "inf")
        0.0
    """
    if isinstance(point, str):
        if point == "inf":
            return _limit_at_infinity(expr, var)
        if point == "-inf":
            return _limit_at_infinity(expr, var, negative=True)
        raise ValueError(f"Invalid limit point: {point}")

    result = _limit_at_point(expr, var, point, direction)

    if result != result:
        raise ValueError(f"Limit does not exist (NaN)")

    return result


def _limit_at_point(
    expr: Expr, var: str, point: float, direction: str
) -> float:
    """Compute limit at a finite point."""
    try:
        result = _evaluate(expr, var, point)
        if result == result:
            return result
    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    result = _lhopital(expr, var, point, direction)
    if result is not None:
        return result

    try:
        eps = 1e-10
        if direction == "right":
            return _evaluate(expr, var, point + eps)
        elif direction == "left":
            return _evaluate(expr, var, point - eps)
        else:
            left_val = _evaluate(expr, var, point - eps)
            right_val = _evaluate(expr, var, point + eps)
            if abs(left_val - right_val) < 1e-6:
                return (left_val + right_val) / 2
    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    raise ValueError(f"Unable to compute limit of {expr} as {var} -> {point}")


def _limit_at_infinity(expr: Expr, var: str, negative: bool = False) -> float:
    """Compute limit as var approaches infinity."""
    sign = -1 if negative else 1
    large = 1e10 * sign

    try:
        return _evaluate(expr, var, large)
    except (ZeroDivisionError, OverflowError):
        pass

    for magnitude in [1e6, 1e8, 1e10, 1e12]:
        try:
            val1 = _evaluate(expr, var, magnitude * sign)
            val2 = _evaluate(expr, var, (magnitude * 2) * sign)
            if abs(val1 - val2) < 1e-6 * max(abs(val1), 1):
                return val1
        except (ZeroDivisionError, OverflowError):
            continue

    raise ValueError(f"Unable to compute limit at infinity")


def _lhopital(
    expr: Expr, var: str, point: float, direction: str
) -> float | None:
    """Apply L'Hôpital's rule for 0/0 or inf/indeterminate forms."""
    if not isinstance(expr, Div):
        return None

    num = expr.numerator
    den = expr.denominator

    try:
        num_val = _evaluate(num, var, point)
        den_val = _evaluate(den, var, point)

        is_zero_zero = abs(num_val) < 1e-10 and abs(den_val) < 1e-10
        is_inf_inf = (
            abs(num_val) > 1e10 and abs(den_val) > 1e10
        ) or (
            (num_val > 1e10 or num_val < -1e10)
            and (den_val > 1e10 or den_val < -1e10)
        )

        if not (is_zero_zero or is_inf_inf):
            return None

        d_num = differentiate(num, var)
        d_den = differentiate(den, var)

        try:
            new_num_val = _evaluate(d_num, var, point)
            new_den_val = _evaluate(d_den, var, point)
            if abs(new_den_val) > 1e-15:
                return new_num_val / new_den_val
        except (ZeroDivisionError, ValueError, OverflowError):
            pass

        result = _lhopital(Div.make(d_num, d_den), var, point, direction)
        return result

    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def _evaluate(expr: Expr, var: str, value: float) -> float:
    """Evaluate expr with var = value."""
    substituted = expr.substitute(var, Constant(value))
    return substituted.evaluate({})


def _try_direct_substitution(expr: Expr, var: str, point: float) -> float | None:
    """Try direct substitution."""
    try:
        return _evaluate(expr, var, point)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
