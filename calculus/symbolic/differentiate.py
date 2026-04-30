from __future__ import annotations

from ..core.expression import (
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
from ..core.simplify import simplify


def differentiate(expr: Expr, var: str, order: int = 1) -> Expr:
    """Compute the symbolic derivative of expr with respect to var.

    Applies standard differentiation rules: power rule, product rule,
    quotient rule, chain rule, and derivatives of all built-in functions.
    The result is automatically simplified.

    Args:
        expr: The expression to differentiate.
        var: The variable name to differentiate with respect to.
        order: The order of the derivative (default 1). Set to 2 for
            the second derivative, 3 for the third, etc.

    Returns:
        The simplified derivative expression.

    Raises:
        TypeError: If the expression type is not supported.

    Examples:
        >>> from calculus import parse, differentiate, pretty
        >>> pretty(differentiate(parse("x^3"), "x"))
        '3 * x ^ 2'
        >>> pretty(differentiate(parse("sin(x)"), "x"))
        'cos(x)'
        >>> pretty(differentiate(parse("x^4"), "x", order=2))
        '12 * x ^ 2'
        >>> pretty(differentiate(parse("sin(x^2)"), "x"))
        '2 * x * cos(x ^ 2)'
    """
    result = expr
    for _ in range(order):
        result = _diff(result, var)
    return simplify(result)


def _diff(expr: Expr, var: str) -> Expr:
    """Compute one step of symbolic differentiation."""
    if isinstance(expr, Constant):
        return Constant(0)

    if isinstance(expr, Symbol):
        return Constant(1) if expr.name == var else Constant(0)

    if isinstance(expr, Neg):
        return Neg.make(_diff(expr.operand, var))

    if isinstance(expr, Add):
        return Add.make(*(_diff(op, var) for op in expr.operands))

    if isinstance(expr, Sub):
        return Sub.make(_diff(expr.left, var), _diff(expr.right, var))

    if isinstance(expr, Mul):
        return _diff_mul(expr, var)

    if isinstance(expr, Div):
        return _diff_div(expr, var)

    if isinstance(expr, Pow):
        return _diff_pow(expr, var)

    if isinstance(expr, Sin):
        return Mul.make(Cos.make(expr.arg), _diff(expr.arg, var))

    if isinstance(expr, Cos):
        return Mul.make(Neg.make(Sin.make(expr.arg)), _diff(expr.arg, var))

    if isinstance(expr, Tan):
        return Mul.make(Sec.make(expr.arg), Sec.make(expr.arg), _diff(expr.arg, var))

    if isinstance(expr, Sec):
        return Mul.make(Sec.make(expr.arg), Tan.make(expr.arg), _diff(expr.arg, var))

    if isinstance(expr, Csc):
        return Mul.make(Neg.make(Csc.make(expr.arg)), Cot.make(expr.arg), _diff(expr.arg, var))

    if isinstance(expr, Cot):
        return Mul.make(Neg.make(Csc.make(expr.arg)), Csc.make(expr.arg), _diff(expr.arg, var))

    if isinstance(expr, Exp):
        return Mul.make(Exp.make(expr.arg), _diff(expr.arg, var))

    if isinstance(expr, Ln):
        return Div.make(_diff(expr.arg, var), expr.arg)

    if isinstance(expr, ArcSin):
        inner = _diff(expr.arg, var)
        return Div.make(
            inner,
            Sqrt.make(Sub.make(Constant(1), Pow.make(expr.arg, Constant(2)))),
        )

    if isinstance(expr, ArcCos):
        inner = _diff(expr.arg, var)
        return Div.make(
            Neg.make(inner),
            Sqrt.make(Sub.make(Constant(1), Pow.make(expr.arg, Constant(2)))),
        )

    if isinstance(expr, ArcTan):
        inner = _diff(expr.arg, var)
        return Div.make(inner, Add.make(Constant(1), Pow.make(expr.arg, Constant(2))))

    if isinstance(expr, Sqrt):
        inner = _diff(expr.arg, var)
        return Div.make(inner, Mul.make(Constant(2), Sqrt.make(expr.arg)))

    if isinstance(expr, Abs):
        inner = _diff(expr.arg, var)
        return Mul.make(Div.make(expr.arg, Abs.make(expr.arg)), inner)

    raise TypeError(f"Cannot differentiate {type(expr).__name__}")


def _diff_mul(expr: Mul, var: str) -> Expr:
    """Apply the product rule: d/dx(f*g) = f'*g + f*g'"""
    operands = expr.operands
    terms: list[Expr] = []

    for i in range(len(operands)):
        factors = list(operands)
        factors[i] = _diff(factors[i], var)
        terms.append(Mul.make(*factors))

    return Add.make(*terms)


def _diff_div(expr: Div, var: str) -> Expr:
    """Apply the quotient rule: d/dx(f/g) = (f'*g - f*g') / g^2"""
    num = expr.numerator
    den = expr.denominator

    dnum = _diff(num, var)
    dden = _diff(den, var)

    numerator = Sub.make(Mul.make(dnum, den), Mul.make(num, dden))
    denominator = Pow.make(den, Constant(2))

    return Div.make(numerator, denominator)


def _diff_pow(expr: Pow, var: str) -> Expr:
    """Apply the general power rule using logarithmic differentiation.

    d/dx(f^g) = f^g * (g' * ln(f) + g * f'/f)
    """
    base = expr.base
    exponent = expr.exponent

    if isinstance(exponent, Constant):
        return Mul.make(
            exponent,
            Pow.make(base, Sub.make(exponent, Constant(1))),
            _diff(base, var),
        )

    if isinstance(base, Constant):
        return Mul.make(
            expr,
            Ln.make(Constant(base.value)),
            _diff(exponent, var),
        )

    d_base = _diff(base, var)
    d_exp = _diff(exponent, var)

    return Mul.make(
        expr,
        Add.make(
            Mul.make(d_exp, Ln.make(base)),
            Mul.make(exponent, Div.make(d_base, base)),
        ),
    )


def partial_derivative(expr: Expr, var: str, order: int = 1) -> Expr:
    """Compute the partial derivative of a multivariable expression.

    This is an alias for ``differentiate()`` provided for clarity when
    working with multivariable expressions. Differentiation with respect
    to a variable treats all other variables as constants.

    Args:
        expr: The expression to differentiate.
        var: The variable name to differentiate with respect to.
        order: The order of the partial derivative (default 1).

    Returns:
        The simplified partial derivative expression.

    Examples:
        >>> from calculus import parse, partial_derivative, pretty
        >>> expr = parse("x^2 * y + y^3")
        >>> pretty(partial_derivative(expr, "x"))
        '2 * x * y'
        >>> pretty(partial_derivative(expr, "y"))
        'x ^ 2 + 3 * y ^ 2'
    """
    return differentiate(expr, var, order)


def gradient(expr: Expr, variables: list[str]) -> list[Expr]:
    """Compute the gradient vector of a scalar field.

    The gradient is the vector of partial derivatives:
    ``[df/dx1, df/dx2, ..., df/dxn]``.

    Args:
        expr: The scalar field expression.
        variables: Ordered list of variable names.

    Returns:
        A list of expressions, one partial derivative per variable.

    Examples:
        >>> from calculus import parse, gradient, pretty
        >>> grad = gradient(parse("x^2 + y^2"), ["x", "y"])
        >>> [pretty(g) for g in grad]
        ['2 * x', '2 * y']
    """
    return [differentiate(expr, var) for var in variables]


def hessian(expr: Expr, variables: list[str]) -> list[list[Expr]]:
    """Compute the Hessian matrix of a scalar field.

    The Hessian is the matrix of second partial derivatives:
    ``H[i][j] = d^2f/(dxi dxj)``.

    Args:
        expr: The scalar field expression.
        variables: Ordered list of variable names.

    Returns:
        A 2D list (n x n) of second derivative expressions.

    Examples:
        >>> from calculus import parse, hessian, pretty
        >>> h = hessian(parse("x^2 * y"), ["x", "y"])
        >>> [[pretty(h[i][j]) for j in range(2)] for i in range(2)]
        [['2 * y', '2 * x'], ['2 * x', '0']]
    """
    grads = gradient(expr, variables)
    return [[differentiate(g, v) for v in variables] for g in grads]
