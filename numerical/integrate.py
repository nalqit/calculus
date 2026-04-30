from __future__ import annotations

from ..core.expression import Constant, Expr


def trapezoidal_rule(
    expr: Expr, var: str, lower: float, upper: float, n: int = 1000
) -> float:
    """Approximate the definite integral using the trapezoidal rule.

    Divides the interval into n subintervals and approximates the area
    under the curve as a sum of trapezoids. Second-order accuracy:
    error is O(1/n²).

    Args:
        expr: The expression.
        var: The variable name.
        lower: Lower bound of integration.
        upper: Upper bound of integration.
        n: Number of subintervals (default 1000). Higher values give
            better accuracy at the cost of performance.

    Returns:
        Approximate integral value.

    Examples:
        >>> from calculus import parse, trapezoidal_rule
        >>> abs(trapezoidal_rule(parse("x^2"), "x", 0, 1, n=10000) - 1/3) < 1e-4
        True
    """
    h = (upper - lower) / n
    result = 0.5 * (f(expr, var, lower) + f(expr, var, upper))
    for i in range(1, n):
        result += f(expr, var, lower + i * h)
    return result * h


def simpsons_rule(
    expr: Expr, var: str, lower: float, upper: float, n: int = 1000
) -> float:
    """Approximate the definite integral using Simpson's rule.

    Uses parabolic arcs instead of straight lines for higher accuracy.
    Fourth-order accuracy: error is O(1/n⁴). The number of subintervals
    is automatically adjusted to be even.

    Args:
        expr: The expression.
        var: The variable name.
        lower: Lower bound of integration.
        upper: Upper bound of integration.
        n: Number of subintervals (default 1000). Will be adjusted to
            be even if necessary.

    Returns:
        Approximate integral value.

    Examples:
        >>> from calculus import parse, simpsons_rule
        >>> abs(simpsons_rule(parse("x^2"), "x", 0, 1) - 1/3) < 1e-10
        True
    """
    if n % 2 != 0:
        n += 1

    h = (upper - lower) / n
    result = f(expr, var, lower) + f(expr, var, upper)

    for i in range(1, n):
        x = lower + i * h
        if i % 2 == 0:
            result += 2 * f(expr, var, x)
        else:
            result += 4 * f(expr, var, x)

    return result * h / 3


def adaptive_quadrature(
    expr: Expr,
    var: str,
    lower: float,
    upper: float,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> float:
    """Approximate the definite integral using adaptive Simpson's quadrature.

    Recursively subdivides the interval, using smaller steps where the
    integrand changes rapidly and larger steps where it is smooth.
    Automatically adjusts to achieve the specified tolerance.

    This is the recommended method for high-accuracy numerical integration.

    Args:
        expr: The expression.
        var: The variable name.
        lower: Lower bound of integration.
        upper: Upper bound of integration.
        tol: Error tolerance (default 1e-10). The algorithm stops when
            the estimated error is below this threshold.
        max_depth: Maximum recursion depth (default 50).

    Returns:
        Approximate integral value with error below ``tol``.

    Examples:
        >>> from calculus import parse, adaptive_quadrature
        >>> abs(adaptive_quadrature(parse("sin(x)"), "x", 0, 3.14159265) - 2) < 1e-8
        True
    """

    def simpson(a: float, b: float) -> float:
        h = (b - a) / 6
        return h * (f(expr, var, a) + 4 * f(expr, var, (a + b) / 2) + f(expr, var, b))

    def _adaptive(a: float, b: float, tol: float, whole: float, depth: int) -> float:
        mid = (a + b) / 2
        left = simpson(a, mid)
        right = simpson(mid, b)
        combined = left + right

        if depth >= max_depth or abs(combined - whole) < 15 * tol:
            return combined + (combined - whole) / 15

        return (
            _adaptive(a, mid, tol / 2, left, depth + 1)
            + _adaptive(mid, b, tol / 2, right, depth + 1)
        )

    whole = simpson(lower, upper)
    return _adaptive(lower, upper, tol, whole, 0)


def gaussian_quadrature(
    expr: Expr, var: str, lower: float, upper: float, n: int = 5
) -> float:
    """Approximate the definite integral using Gaussian quadrature.

    Evaluates the integrand at optimally chosen points (Gauss-Legendre
    nodes) for maximum accuracy with minimal function evaluations.
    Exact for polynomials of degree up to ``2n-1``.

    Very efficient for smooth integrands. Supports 1 to 5 quadrature points.

    Args:
        expr: The expression.
        var: The variable name.
        lower: Lower bound of integration.
        upper: Upper bound of integration.
        n: Number of quadrature points (1-5, default 5).

    Returns:
        Approximate integral value.

    Raises:
        ValueError: If n is not between 1 and 5.

    Examples:
        >>> from calculus import parse, gaussian_quadrature
        >>> abs(gaussian_quadrature(parse("x^2"), "x", 0, 1, n=3) - 1/3) < 1e-10
        True
    """
    nodes_weights = {
        1: [(0.0, 2.0)],
        2: [(-0.5773502692, 1.0), (0.5773502692, 1.0)],
        3: [
            (-0.7745966692, 0.5555555556),
            (0.0, 0.8888888889),
            (0.7745966692, 0.5555555556),
        ],
        4: [
            (-0.8611363116, 0.3478548451),
            (-0.3399810436, 0.6521451549),
            (0.3399810436, 0.6521451549),
            (0.8611363116, 0.3478548451),
        ],
        5: [
            (-0.9061798459, 0.2369268850),
            (-0.5384693101, 0.4786286705),
            (0.0, 0.5688888889),
            (0.5384693101, 0.4786286705),
            (0.9061798459, 0.2369268850),
        ],
    }

    if n not in nodes_weights:
        raise ValueError(f"Unsupported number of points: {n}. Use 1-5.")

    mid = (upper + lower) / 2
    half_range = (upper - lower) / 2

    result = 0.0
    for node, weight in nodes_weights[n]:
        x = mid + half_range * node
        result += weight * f(expr, var, x)

    return result * half_range


def f(expr: Expr, var: str, value: float) -> float:
    """Evaluate expr with var = value."""
    substituted = expr.substitute(var, Constant(value))
    return substituted.evaluate({})
