from __future__ import annotations

from typing import TYPE_CHECKING

from .expression import (
    Abs,
    Add,
    ArcCos,
    ArcSin,
    ArcTan,
    Constant,
    Cos,
    Cot,
    Csc,
    Div,
    Exp,
    Expr,
    Function,
    Ln,
    Mul,
    Neg,
    Pow,
    Sec,
    Sin,
    Sqrt,
    Sub,
    Symbol,
    Tan,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_PRECEDENCE: dict[type[Expr], int] = {
    Add: 1,
    Sub: 1,
    Mul: 2,
    Div: 2,
    Pow: 3,
    Neg: 4,
}

_PARENTHESE_NEEDED: dict[tuple[type[Expr], type[Expr]], bool] = {}


def pretty(expr: Expr) -> str:
    """Format an expression as a human-readable ASCII string.

    Handles operator precedence correctly, adding parentheses only where needed.
    Uses ``*`` for multiplication and ``^`` for exponentiation.

    Args:
        expr: The expression to format.

    Returns:
        A human-readable string representation.

    Examples:
        >>> from calcus import parse, pretty
        >>> pretty(parse("x^2 + 2*x + 1"))
        'x ^ 2 + 2 * x + 1'
    """
    return _format(expr, parent_precedence=0)


def _format(expr: Expr, parent_precedence: int = 0) -> str:
    precedence = _PRECEDENCE.get(type(expr), 5)
    needs_parens = precedence < parent_precedence

    result = _format_inner(expr, precedence)

    if needs_parens:
        return f"({result})"
    return result


def _format_inner(expr: Expr, precedence: int) -> str:
    if isinstance(expr, Constant):
        if expr.value == int(expr.value):
            return str(int(expr.value))
        return str(expr.value)

    if isinstance(expr, Symbol):
        return expr.name

    if isinstance(expr, Add):
        parts = [_format(expr.operands[0], precedence)]
        for op in expr.operands[1:]:
            parts.append(f" + {_format(op, precedence)}")
        return "".join(parts)

    if isinstance(expr, Sub):
        return f"{_format(expr.left, precedence)} - {_format(expr.right, precedence + 1)}"

    if isinstance(expr, Mul):
        parts = []
        for op in expr.operands:
            formatted = _format(op, precedence)
            parts.append(formatted)
        return " * ".join(parts)

    if isinstance(expr, Div):
        num = _format(expr.numerator, precedence)
        den = _format(expr.denominator, precedence + 1)
        return f"{num} / {den}"

    if isinstance(expr, Pow):
        base = _format(expr.base, precedence + 1)
        exp = _format(expr.exponent, precedence + 1)
        return f"{base} ^ {exp}"

    if isinstance(expr, Neg):
        operand = _format(expr.operand, precedence + 1)
        return f"-{operand}"

    if isinstance(expr, Function):
        return f"{expr.name}({pretty(expr.arg)})"

    return repr(expr)


def to_latex(expr: Expr) -> str:
    """Format an expression as a LaTeX string.

    Produces LaTeX suitable for math rendering. Uses ``\\cdot`` for
    multiplication, ``\\frac{}{}`` for division, and standard LaTeX
    function names (``\\sin``, ``\\exp``, ``\\sqrt``, etc.).

    Args:
        expr: The expression to format.

    Returns:
        A LaTeX string representation of the expression.

    Examples:
        >>> from calcus import parse, to_latex
        >>> to_latex(parse("x^2 + 1"))
        'x^{2} + 1'
        >>> to_latex(parse("sin(x) / x"))
        '\\\\frac{\\\\sin(x)}{x}'
    """
    return _latex(expr, parent_precedence=0)


def _latex(expr: Expr, parent_precedence: int = 0) -> str:
    precedence = _PRECEDENCE.get(type(expr), 5)
    needs_parens = precedence < parent_precedence
    result = _latex_inner(expr, precedence)
    if needs_parens:
        return rf"\left({result}\right)"
    return result


def _latex_inner(expr: Expr, precedence: int) -> str:
    if isinstance(expr, Constant):
        if expr.value == int(expr.value):
            return str(int(expr.value))
        return str(expr.value)

    if isinstance(expr, Symbol):
        return expr.name

    if isinstance(expr, Add):
        parts = [_latex(expr.operands[0], precedence)]
        for op in expr.operands[1:]:
            parts.append(f" + {_latex(op, precedence)}")
        return "".join(parts)

    if isinstance(expr, Sub):
        return f"{_latex(expr.left, precedence)} - {_latex(expr.right, precedence + 1)}"

    if isinstance(expr, Mul):
        parts = []
        for op in expr.operands:
            parts.append(_latex(op, precedence))
        return " \\cdot ".join(parts)

    if isinstance(expr, Div):
        num = _latex(expr.numerator, 0)
        den = _latex(expr.denominator, 0)
        return rf"\frac{{{num}}}{{{den}}}"

    if isinstance(expr, Pow):
        base = _latex(expr.base, precedence + 1)
        exp = _latex(expr.exponent, precedence + 1)
        return f"{base}^{{{exp}}}"

    if isinstance(expr, Neg):
        return f"-{_latex(expr.operand, precedence + 1)}"

    latex_functions: dict[type[Function], str | Callable[[Expr], str]] = {
        Sin: (r"\sin", lambda e: f"\\sin({_latex(e.arg, 0)})"),
        Cos: (r"\cos", lambda e: f"\\cos({_latex(e.arg, 0)})"),
        Tan: (r"\tan", lambda e: f"\\tan({_latex(e.arg, 0)})"),
        Exp: (r"\exp", lambda e: f"e^{{{_latex(e.arg, 0)}}}"),
        Ln: (r"\ln", lambda e: f"\\ln({_latex(e.arg, 0)})"),
        Sqrt: (r"\sqrt", lambda e: f"\\sqrt{{{_latex(e.arg, 0)}}}"),
        Abs: (r"\lvert", lambda e: f"\\left\\lvert {_latex(e.arg, 0)} \\right\\rvert"),
        ArcSin: (r"\arcsin", lambda e: f"\\arcsin({_latex(e.arg, 0)})"),
        ArcCos: (r"\arccos", lambda e: f"\\arccos({_latex(e.arg, 0)})"),
        ArcTan: (r"\arctan", lambda e: f"\\arctan({_latex(e.arg, 0)})"),
        Sec: (r"\sec", lambda e: f"\\sec({_latex(e.arg, 0)})"),
        Csc: (r"\csc", lambda e: f"\\csc({_latex(e.arg, 0)})"),
        Cot: (r"\cot", lambda e: f"\\cot({_latex(e.arg, 0)})"),
    }

    for func_type, (_, formatter) in latex_functions.items():
        if isinstance(expr, func_type):
            return formatter(expr)

    return expr.name
