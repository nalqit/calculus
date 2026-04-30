from __future__ import annotations

from ..core.expression import Constant, Expr


def numerical_diff(
    expr: Expr, var: str, point: float, h: float = 1e-7, method: str = "central"
) -> float:
    """Approximate the derivative of expr at a point using finite differences.

    Three methods are available:
    - ``forward``: ``[f(x+h) - f(x)] / h`` (first-order accuracy)
    - ``backward``: ``[f(x) - f(x-h)] / h`` (first-order accuracy)
    - ``central``: ``[f(x+h) - f(x-h)] / (2h)`` (second-order accuracy, default)

    The central difference method is recommended for most use cases as it
    provides better accuracy.

    Args:
        expr: The expression.
        var: The variable name.
        point: The point at which to evaluate the derivative.
        h: Step size (default 1e-7). Smaller values give better accuracy
            up to the limits of floating-point precision.
        method: ``"forward"``, ``"backward"``, or ``"central"`` (default).

    Returns:
        Approximate derivative value at the given point.

    Raises:
        ValueError: If an unknown method is specified.

    Examples:
        >>> from calculus import parse, numerical_diff
        >>> abs(numerical_diff(parse("x^2"), "x", 2) - 4) < 1e-5
        True
        >>> abs(numerical_diff(parse("sin(x)"), "x", 0) - 1) < 1e-5
        True
    """
    if method == "forward":
        return (f(expr, var, point + h) - f(expr, var, point)) / h
    elif method == "backward":
        return (f(expr, var, point) - f(expr, var, point - h)) / h
    elif method == "central":
        return (f(expr, var, point + h) - f(expr, var, point - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'forward', 'backward', or 'central'")


def numerical_diff2(
    expr: Expr, var: str, point: float, h: float = 1e-5
) -> float:
    """Approximate the second derivative of expr at a point.

    Uses the central difference formula:
    ``f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²``

    Args:
        expr: The expression.
        var: The variable name.
        point: The point at which to evaluate the second derivative.
        h: Step size (default 1e-5).

    Returns:
        Approximate second derivative value.

    Examples:
        >>> from calculus import parse, numerical_diff2
        >>> abs(numerical_diff2(parse("x^3"), "x", 2) - 12) < 1e-3
        True
    """
    return (f(expr, var, point + h) - 2 * f(expr, var, point) + f(expr, var, point - h)) / (h * h)


def f(expr: Expr, var: str, value: float) -> float:
    """Evaluate expr with var = value."""
    substituted = expr.substitute(var, Constant(value))
    return substituted.evaluate({})
