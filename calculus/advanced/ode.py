from __future__ import annotations

from ..core.expression import (
    Add,
    Constant,
    Expr,
    Mul,
    Symbol,
)
from ..core.simplify import simplify
from ..symbolic.differentiate import differentiate


class ODESolver:
    """Numerical solvers for ordinary differential equations.

    Solves initial value problems of the form:
    ``dy/dx = f(x, y),  y(x0) = y0``
    """

    @staticmethod
    def euler(
        f_func,
        y0: float,
        x0: float,
        x_end: float,
        h: float = 0.01,
    ) -> list[tuple[float, float]]:
        """Solve dy/dx = f(x, y) using Euler's method.

        The simplest ODE solver. First-order accurate: error is O(h).
        Use a small step size for acceptable accuracy.

        Args:
            f_func: A callable ``f(x, y)`` returning dy/dx.
            y0: Initial condition y(x0).
            x0: Initial x value.
            x_end: Final x value.
            h: Step size (default 0.01).

        Returns:
            A list of ``(x, y)`` tuples from x0 to x_end.

        Examples:
            >>> results = ODESolver.euler(lambda x, y: y, 1.0, 0.0, 0.1, h=0.1)
            >>> len(results)
            2
        """
        results = [(x0, y0)]
        x, y = x0, y0

        while x < x_end:
            y = y + h * f_func(x, y)
            x = x + h
            results.append((x, y))

        return results

    @staticmethod
    def runge_kutta_4(
        f_func,
        y0: float,
        x0: float,
        x_end: float,
        h: float = 0.01,
    ) -> list[tuple[float, float]]:
        """Solve dy/dx = f(x, y) using 4th-order Runge-Kutta method.

        The most commonly used ODE solver. Fourth-order accurate: error is O(h⁴).
        Provides much better accuracy than Euler's method for the same step size.

        Args:
            f_func: A callable ``f(x, y)`` returning dy/dx.
            y0: Initial condition y(x0).
            x0: Initial x value.
            x_end: Final x value.
            h: Step size (default 0.01).

        Returns:
            A list of ``(x, y)`` tuples from x0 to x_end.

        Examples:
            >>> # dy/dx = y, y(0) = 1 => y = e^x
            >>> results = ODESolver.runge_kutta_4(lambda x, y: y, 1.0, 0.0, 1.0, h=0.1)
            >>> abs(results[-1][1] - 2.71828) < 1e-4  # y(1) ≈ e
            True
        """
        results = [(x0, y0)]
        x, y = x0, y0

        while x < x_end:
            k1 = f_func(x, y)
            k2 = f_func(x + h / 2, y + h * k1 / 2)
            k3 = f_func(x + h / 2, y + h * k2 / 2)
            k4 = f_func(x + h, y + h * k3)

            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x + h
            results.append((x, y))

        return results
