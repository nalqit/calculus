from __future__ import annotations

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


def simplify(expr: Expr) -> Expr:
    """Recursively simplify an expression tree.

    Applies algebraic simplification rules until no more changes can be made:
    - Constant folding: ``2 + 3`` → ``5``, ``2 * x`` stays, ``2 * 3`` → ``6``
    - Identity elimination: ``x + 0`` → ``x``, ``x * 1`` → ``x``, ``x * 0`` → ``0``
    - Power simplification: ``x^0`` → ``1``, ``x^1`` → ``x``
    - Inverse cancellation: ``x * (1/x)`` → ``1``, ``x^2 / x`` → ``x``
    - Self-subtraction: ``x - x`` → ``0``, ``x / x`` → ``1``
    - Nested power simplification when the exponent can be reduced.

    Args:
        expr: The expression to simplify.

    Returns:
        A simplified expression. May be equal to ``expr`` if no simplifications apply.

    Examples:
        >>> from calcus import parse, simplify, pretty
        >>> pretty(simplify(parse("x + 0")))
        'x'
        >>> pretty(simplify(parse("x * 1")))
        'x'
        >>> pretty(simplify(parse("x ^ 0")))
        '1'
    """
    simplified = _simplify_inner(expr)

    if simplified != expr:
        return simplify(simplified)

    return simplified


def _simplify_inner(expr: Expr) -> Expr:
    """One pass of simplification."""
    if isinstance(expr, Constant):
        return expr

    if isinstance(expr, Symbol):
        return expr

    if isinstance(expr, Add):
        new_operands = [_simplify_inner(op) for op in expr.operands]
        return _simplify_add(new_operands)

    if isinstance(expr, Mul):
        new_operands = [_simplify_inner(op) for op in expr.operands]
        return _simplify_mul(new_operands)

    if isinstance(expr, Pow):
        new_base = _simplify_inner(expr.base)
        new_exp = _simplify_inner(expr.exponent)
        return _simplify_pow(new_base, new_exp)

    if isinstance(expr, Sub):
        new_left = _simplify_inner(expr.left)
        new_right = _simplify_inner(expr.right)
        return _simplify_sub(new_left, new_right)

    if isinstance(expr, Div):
        new_num = _simplify_inner(expr.numerator)
        new_den = _simplify_inner(expr.denominator)
        return _simplify_div(new_num, new_den)

    if isinstance(expr, Neg):
        new_operand = _simplify_inner(expr.operand)
        return Neg.make(new_operand)

    if isinstance(expr, Function):
        new_arg = _simplify_inner(expr.arg)
        return type(expr).make(new_arg)

    return expr


def _simplify_add(operands: list[Expr]) -> Expr:
    """Simplify addition by combining constants and like terms."""
    const_sum = 0.0
    new_ops: list[Expr] = []

    for op in operands:
        if isinstance(op, Constant):
            const_sum += op.value
        else:
            new_ops.append(op)

    if const_sum != 0:
        new_ops.insert(0, Constant(const_sum))

    if not new_ops:
        return Constant(0)
    if len(new_ops) == 1:
        return new_ops[0]

    return Add(*new_ops)


def _simplify_mul(operands: list[Expr]) -> Expr:
    """Simplify multiplication by combining constants and canceling terms."""
    const_prod = 1.0
    new_ops: list[Expr] = []
    has_zero = False

    for op in operands:
        if isinstance(op, Constant):
            if op.value == 0:
                has_zero = True
            const_prod *= op.value
        else:
            new_ops.append(op)

    if has_zero:
        return Constant(0)

    if const_prod == 0:
        return Constant(0)

    new_ops = _cancel_inverses(new_ops)

    if const_prod != 1.0:
        new_ops.insert(0, Constant(const_prod))

    if not new_ops:
        return Constant(1)
    if len(new_ops) == 1:
        return new_ops[0]

    return Mul(*new_ops)


def _cancel_inverses(operands: list[Expr]) -> list[Expr]:
    """Cancel out terms like x * (1/x) or x^a / x = x^(a-1)."""
    result = list(operands)
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(result):
            op = result[i]
            if isinstance(op, Div) and isinstance(op.numerator, Constant) and op.numerator.value == 1:
                denom = op.denominator
                for j in range(len(result)):
                    if i == j:
                        continue
                    candidate = result[j]

                    if candidate == denom:
                        result.pop(max(i, j))
                        result.pop(min(i, j))
                        changed = True
                        i = -1
                        break

                    if isinstance(candidate, Pow) and candidate.base == denom:
                        if isinstance(candidate.exponent, Constant):
                            new_exp = Constant(candidate.exponent.value - 1)
                            if new_exp.value == 1:
                                result[j] = candidate.base
                            elif new_exp.value == 0:
                                result.pop(j)
                            else:
                                result[j] = Pow(candidate.base, new_exp)
                            result.pop(i)
                            changed = True
                            i = -1
                            break
            i += 1

    return result


def _simplify_pow(base: Expr, exponent: Expr) -> Expr:
    """Simplify power expressions."""
    if isinstance(base, Constant) and isinstance(exponent, Constant):
        return Constant(base.value ** exponent.value)

    if isinstance(exponent, Constant):
        if exponent.value == 0:
            return Constant(1)
        if exponent.value == 1:
            return base
        if exponent.value == -1:
            return Div.make(Constant(1), base)

    if isinstance(base, Constant):
        if base.value == 0:
            return Constant(0)
        if base.value == 1:
            return Constant(1)

    if isinstance(exponent, Sub):
        simplified_exp = simplify(exponent)
        if isinstance(simplified_exp, Constant):
            return _simplify_pow(base, simplified_exp)

    return Pow(base, exponent)


def _simplify_sub(left: Expr, right: Expr) -> Expr:
    """Simplify subtraction."""
    if isinstance(right, Constant) and right.value == 0:
        return left
    if isinstance(left, Constant) and left.value == 0:
        return Neg.make(right)
    if left == right:
        return Constant(0)
    if isinstance(left, Constant) and isinstance(right, Constant):
        return Constant(left.value - right.value)
    return Sub(left, right)


def _simplify_div(numerator: Expr, denominator: Expr) -> Expr:
    """Simplify division."""
    if isinstance(numerator, Constant) and numerator.value == 0:
        return Constant(0)
    if isinstance(denominator, Constant) and denominator.value == 1:
        return numerator
    if denominator == numerator:
        return Constant(1)
    if isinstance(denominator, Constant) and denominator.value != 0 and numerator == denominator:
        return Constant(1)
    return Div(numerator, denominator)
