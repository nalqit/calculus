from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from math import prod as math_prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Expr(ABC):
    """Base class for all mathematical expression nodes."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, bindings: dict[str, float]) -> float:
        """Evaluate the expression with given variable bindings."""
        ...

    @abstractmethod
    def symbols(self) -> set[str]:
        """Return all variable names in this expression."""
        ...

    @abstractmethod
    def substitute(self, var: str, replacement: Expr) -> Expr:
        """Replace all occurrences of var with replacement."""
        ...

    def __add__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Add(self, other)

    def __radd__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Add(other, self)

    def __sub__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Sub(self, other)

    def __rsub__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Sub(other, self)

    def __mul__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Mul(self, other)

    def __rmul__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Mul(other, self)

    def __truediv__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Div(self, other)

    def __rtruediv__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Div(other, self)

    def __pow__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Pow(self, other)

    def __rpow__(self, other: Expr | int | float) -> Expr:
        if isinstance(other, (int, float)):
            other = Constant(other)
        return Pow(other, self)

    def __neg__(self) -> Expr:
        return Neg(self)

    def __pos__(self) -> Expr:
        return self


@dataclass(frozen=True)
class Constant(Expr):
    """A numeric constant."""

    value: float

    def __repr__(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("Constant", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constant) and self.value == other.value

    def evaluate(self, bindings: dict[str, float]) -> float:
        return self.value

    def symbols(self) -> set[str]:
        return set()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return self

    @cached_property
    def is_zero(self) -> bool:
        return self.value == 0

    @cached_property
    def is_one(self) -> bool:
        return self.value == 1

    @cached_property
    def is_negative_one(self) -> bool:
        return self.value == -1

    @cached_property
    def is_integer(self) -> bool:
        return self.value == int(self.value)


@dataclass(frozen=True)
class Symbol(Expr):
    """A variable symbol."""

    name: str

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("Symbol", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symbol) and self.name == other.name

    def evaluate(self, bindings: dict[str, float]) -> float:
        if self.name not in bindings:
            raise ValueError(f"No value provided for symbol '{self.name}'")
        return bindings[self.name]

    def symbols(self) -> set[str]:
        return {self.name}

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return replacement if self.name == var else self


@dataclass(frozen=True)
class Add(Expr):
    """Addition of two or more expressions."""

    operands: tuple[Expr, ...]

    def __init__(self, *operands: Expr):
        object.__setattr__(self, "operands", tuple(operands))

    def __post_init__(self) -> None:
        pass

    @staticmethod
    def make(*operands: Expr) -> Expr:
        """Create an Add, simplifying where possible."""
        flattened: list[Expr] = []
        for op in operands:
            if isinstance(op, Add):
                flattened.extend(op.operands)
            elif isinstance(op, Constant) and op.value == 0:
                continue
            else:
                flattened.append(op)

        if not flattened:
            return Constant(0)
        if len(flattened) == 1:
            return flattened[0]
        return Add(*flattened)

    def __repr__(self) -> str:
        return f"Add({', '.join(repr(o) for o in self.operands)})"

    def __hash__(self) -> int:
        return hash(("Add", self.operands))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Add) and self.operands == other.operands

    def evaluate(self, bindings: dict[str, float]) -> float:
        return sum(op.evaluate(bindings) for op in self.operands)

    def symbols(self) -> set[str]:
        result: set[str] = set()
        for op in self.operands:
            result.update(op.symbols())
        return result

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return Add.make(*(op.substitute(var, replacement) for op in self.operands))


@dataclass(frozen=True)
class Mul(Expr):
    """Multiplication of two or more expressions."""

    operands: tuple[Expr, ...]

    def __init__(self, *operands: Expr):
        object.__setattr__(self, "operands", tuple(operands))

    @staticmethod
    def make(*operands: Expr) -> Expr:
        """Create a Mul, simplifying where possible."""
        flattened: list[Expr] = []
        coeff = 1.0

        for op in operands:
            if isinstance(op, Mul):
                flattened.extend(op.operands)
            elif isinstance(op, Constant):
                if op.value == 0:
                    return Constant(0)
                coeff *= op.value
            else:
                flattened.append(op)

        if not flattened:
            return Constant(coeff)

        if coeff == 0:
            return Constant(0)

        if coeff != 1.0:
            flattened.insert(0, Constant(coeff))

        if len(flattened) == 1:
            return flattened[0]

        return Mul(*flattened)

    def __repr__(self) -> str:
        return f"Mul({', '.join(repr(o) for o in self.operands)})"

    def __hash__(self) -> int:
        return hash(("Mul", self.operands))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mul) and self.operands == other.operands

    def evaluate(self, bindings: dict[str, float]) -> float:
        result = 1.0
        for op in self.operands:
            result *= op.evaluate(bindings)
        return result

    def symbols(self) -> set[str]:
        result: set[str] = set()
        for op in self.operands:
            result.update(op.symbols())
        return result

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return Mul.make(*(op.substitute(var, replacement) for op in self.operands))


@dataclass(frozen=True)
class Pow(Expr):
    """Exponentiation."""

    base: Expr
    exponent: Expr

    @staticmethod
    def make(base: Expr, exponent: Expr) -> Expr:
        """Create a Pow, simplifying where possible."""
        if isinstance(exponent, Constant) and exponent.value == 0:
            return Constant(1)
        if isinstance(exponent, Constant) and exponent.value == 1:
            return base
        if isinstance(base, Constant) and base.value == 0:
            return Constant(0)
        if isinstance(base, Constant) and base.value == 1:
            return Constant(1)
        return Pow(base, exponent)

    def __repr__(self) -> str:
        return f"Pow({self.base!r}, {self.exponent!r})"

    def __hash__(self) -> int:
        return hash(("Pow", self.base, self.exponent))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Pow)
            and self.base == other.base
            and self.exponent == other.exponent
        )

    def evaluate(self, bindings: dict[str, float]) -> float:
        return self.base.evaluate(bindings) ** self.exponent.evaluate(bindings)

    def symbols(self) -> set[str]:
        return self.base.symbols() | self.exponent.symbols()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return Pow.make(
            self.base.substitute(var, replacement),
            self.exponent.substitute(var, replacement),
        )


@dataclass(frozen=True)
class Sub(Expr):
    """Subtraction (represented as addition of a negation)."""

    left: Expr
    right: Expr

    @staticmethod
    def make(left: Expr, right: Expr) -> Expr:
        """Create a Sub, simplifying where possible."""
        if isinstance(right, Constant) and right.value == 0:
            return left
        if isinstance(left, Constant) and left.value == 0:
            return Neg.make(right)
        return Sub(left, right)

    def __repr__(self) -> str:
        return f"Sub({self.left!r}, {self.right!r})"

    def __hash__(self) -> int:
        return hash(("Sub", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sub)
            and self.left == other.left
            and self.right == other.right
        )

    def evaluate(self, bindings: dict[str, float]) -> float:
        return self.left.evaluate(bindings) - self.right.evaluate(bindings)

    def symbols(self) -> set[str]:
        return self.left.symbols() | self.right.symbols()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return Sub.make(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement),
        )


@dataclass(frozen=True)
class Div(Expr):
    """Division (represented as multiplication by reciprocal)."""

    numerator: Expr
    denominator: Expr

    @staticmethod
    def make(numerator: Expr, denominator: Expr) -> Expr:
        """Create a Div, simplifying where possible."""
        if isinstance(denominator, Constant) and denominator.value == 0:
            raise ZeroDivisionError("Division by zero in expression")
        if isinstance(numerator, Constant) and numerator.value == 0:
            return Constant(0)
        if isinstance(denominator, Constant) and denominator.value == 1:
            return numerator
        return Div(numerator, denominator)

    def __repr__(self) -> str:
        return f"Div({self.numerator!r}, {self.denominator!r})"

    def __hash__(self) -> int:
        return hash(("Div", self.numerator, self.denominator))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Div)
            and self.numerator == other.numerator
            and self.denominator == other.denominator
        )

    def evaluate(self, bindings: dict[str, float]) -> float:
        return self.numerator.evaluate(bindings) / self.denominator.evaluate(bindings)

    def symbols(self) -> set[str]:
        return self.numerator.symbols() | self.denominator.symbols()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return Div.make(
            self.numerator.substitute(var, replacement),
            self.denominator.substitute(var, replacement),
        )


@dataclass(frozen=True)
class Neg(Expr):
    """Negation of an expression."""

    operand: Expr

    @staticmethod
    def make(operand: Expr) -> Expr:
        """Create a Neg, simplifying where possible."""
        if isinstance(operand, Neg):
            return operand.operand
        if isinstance(operand, Constant):
            return Constant(-operand.value)
        return Neg(operand)

    def __repr__(self) -> str:
        return f"Neg({self.operand!r})"

    def __hash__(self) -> int:
        return hash(("Neg", self.operand))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Neg) and self.operand == other.operand

    def evaluate(self, bindings: dict[str, float]) -> float:
        return -self.operand.evaluate(bindings)

    def symbols(self) -> set[str]:
        return self.operand.symbols()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return Neg.make(self.operand.substitute(var, replacement))


class Function(Expr):
    """Base class for mathematical functions (sin, cos, exp, log, etc.)."""

    def __init__(self, arg: Expr):
        self.arg = arg

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def make(cls, arg: Expr) -> Function:
        ...

    def __repr__(self) -> str:
        return f"{self.name}({self.arg!r})"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.arg))

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.arg == other.arg

    def evaluate(self, bindings: dict[str, float]) -> float:
        return self._eval(self.arg.evaluate(bindings))

    @abstractmethod
    def _eval(self, x: float) -> float:
        ...

    def symbols(self) -> set[str]:
        return self.arg.symbols()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return self.make(self.arg.substitute(var, replacement))


class Sin(Function):
    @property
    def name(self) -> str:
        return "sin"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import sin

            return Constant(sin(arg.value))
        return Sin(arg)

    def _eval(self, x: float) -> float:
        from math import sin

        return sin(x)


class Cos(Function):
    @property
    def name(self) -> str:
        return "cos"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import cos

            return Constant(cos(arg.value))
        return Cos(arg)

    def _eval(self, x: float) -> float:
        from math import cos

        return cos(x)


class Tan(Function):
    @property
    def name(self) -> str:
        return "tan"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import tan

            return Constant(tan(arg.value))
        return Tan(arg)

    def _eval(self, x: float) -> float:
        from math import tan

        return tan(x)


class Exp(Function):
    @property
    def name(self) -> str:
        return "exp"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import exp

            return Constant(exp(arg.value))
        return Exp(arg)

    def _eval(self, x: float) -> float:
        from math import exp

        return exp(x)


class Ln(Function):
    @property
    def name(self) -> str:
        return "ln"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import log

            return Constant(log(arg.value))
        return Ln(arg)

    def _eval(self, x: float) -> float:
        from math import log

        return log(x)


class Sqrt(Function):
    @property
    def name(self) -> str:
        return "sqrt"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import sqrt

            return Constant(sqrt(arg.value))
        return Sqrt(arg)

    def _eval(self, x: float) -> float:
        from math import sqrt

        return sqrt(x)


class Abs(Function):
    @property
    def name(self) -> str:
        return "abs"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            return Constant(abs(arg.value))
        return Abs(arg)

    def _eval(self, x: float) -> float:
        return abs(x)


class ArcSin(Function):
    @property
    def name(self) -> str:
        return "asin"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import asin

            return Constant(asin(arg.value))
        return ArcSin(arg)

    def _eval(self, x: float) -> float:
        from math import asin

        return asin(x)


class ArcCos(Function):
    @property
    def name(self) -> str:
        return "acos"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import acos

            return Constant(acos(arg.value))
        return ArcCos(arg)

    def _eval(self, x: float) -> float:
        from math import acos

        return acos(x)


class ArcTan(Function):
    @property
    def name(self) -> str:
        return "atan"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import atan

            return Constant(atan(arg.value))
        return ArcTan(arg)

    def _eval(self, x: float) -> float:
        from math import atan

        return atan(x)


class Sec(Function):
    @property
    def name(self) -> str:
        return "sec"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        return Sec(arg)

    def _eval(self, x: float) -> float:
        from math import cos

        return 1 / cos(x)


class Csc(Function):
    @property
    def name(self) -> str:
        return "csc"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        return Csc(arg)

    def _eval(self, x: float) -> float:
        from math import sin

        return 1 / sin(x)


class Cot(Function):
    @property
    def name(self) -> str:
        return "cot"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        return Cot(arg)

    def _eval(self, x: float) -> float:
        from math import tan

        return 1 / tan(x)


def constant(value: int | float) -> Constant:
    return Constant(float(value))


def symbol(name: str) -> Symbol:
    return Symbol(name)


# Common constants
ZERO = Constant(0)
ONE = Constant(1)
NEG_ONE = Constant(-1)
