from __future__ import annotations

import re
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
    from collections.abc import Sequence


FUNCTION_MAP: dict[str, type[Function]] = {
    "sin": Sin,
    "cos": Cos,
    "tan": Tan,
    "exp": Exp,
    "ln": Ln,
    "log": Ln,
    "sqrt": Sqrt,
    "abs": Abs,
    "asin": ArcSin,
    "acos": ArcCos,
    "atan": ArcTan,
    "arcsin": ArcSin,
    "arccos": ArcCos,
    "arctan": ArcTan,
    "sec": Sec,
    "csc": Csc,
    "cot": Cot,
}


class ParseError(Exception):
    """Raised when the parser encounters an error."""


class Token:
    __slots__ = ("kind", "value", "pos")

    def __init__(self, kind: str, value: str, pos: int):
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, pos={self.pos})"


def tokenize(source: str) -> list[Token]:
    """Convert source string into a list of tokens."""
    tokens: list[Token] = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        if ch in " \t\n\r":
            i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            start = i
            while i < n and (source[i].isdigit() or source[i] == "."):
                i += 1
            tokens.append(Token("NUMBER", source[start:i], start))
            continue

        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            if word in FUNCTION_MAP:
                tokens.append(Token("FUNC", word, start))
            else:
                tokens.append(Token("IDENT", word, start))
            continue

        op_map = {
            "+": "PLUS",
            "-": "MINUS",
            "*": "STAR",
            "/": "SLASH",
            "^": "CARET",
            "(": "LPAREN",
            ")": "RPAREN",
            ",": "COMMA",
        }
        if ch in op_map:
            tokens.append(Token(op_map[ch], ch, i))
            i += 1
            continue

        raise ParseError(f"Unexpected character '{ch}' at position {i}")

    tokens.append(Token("EOF", "", n))
    return tokens


class Parser:
    """Recursive descent parser for mathematical expressions.

    Grammar:
        expr       -> term (('+' | '-') term)*
        term       -> power (('*' | '/') power)*
        power      -> unary ('^' power)?    (right-associative)
        unary      -> ('-' | '+') unary | call
        call       -> primary ('(' expr ')')?
        primary    -> NUMBER | IDENT | FUNC '(' expr ')' | '(' expr ')'
    """

    def __init__(self, source: str):
        self.tokens = tokenize(source)
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, kind: str) -> bool:
        return self.current.kind == kind

    def advance(self) -> Token:
        tok = self.current
        self.pos += 1
        return tok

    def expect(self, kind: str) -> Token:
        if not self.peek(kind):
            raise ParseError(
                f"Expected {kind}, got {self.current.kind} at pos {self.current.pos}"
            )
        return self.advance()

    def parse(self) -> Expr:
        expr = self.parse_expr()
        if not self.peek("EOF"):
            raise ParseError(f"Unexpected token: {self.current}")
        return expr

    def parse_expr(self) -> Expr:
        """expr -> term (('+' | '-') term)*"""
        left = self.parse_term()

        while self.peek("PLUS") or self.peek("MINUS"):
            op = self.advance()
            right = self.parse_term()
            if op.kind == "PLUS":
                left = Add.make(left, right)
            else:
                left = Sub.make(left, right)

        return left

    def parse_term(self) -> Expr:
        """term -> power (('*' | '/') power)*"""
        left = self.parse_power()

        while self.peek("STAR") or self.peek("SLASH"):
            op = self.advance()
            right = self.parse_power()
            if op.kind == "STAR":
                left = Mul.make(left, right)
            else:
                left = Div.make(left, right)

        return left

    def parse_power(self) -> Expr:
        """power -> unary ('^' power)?  (right-associative)"""
        base = self.parse_unary()

        if self.peek("CARET"):
            self.advance()
            exponent = self.parse_power()
            return Pow.make(base, exponent)

        return base

    def parse_unary(self) -> Expr:
        """unary -> ('-' | '+') unary | call"""
        if self.peek("MINUS"):
            self.advance()
            operand = self.parse_unary()
            return Neg.make(operand)

        if self.peek("PLUS"):
            self.advance()
            return self.parse_unary()

        return self.parse_call()

    def parse_call(self) -> Expr:
        """call -> primary ('(' expr ')')?"""
        primary = self.parse_primary()

        if self.peek("LPAREN"):
            self.advance()
            arg = self.parse_expr()
            self.expect("RPAREN")

            if isinstance(primary, Symbol):
                func_name = primary.name
                if func_name in FUNCTION_MAP:
                    return FUNCTION_MAP[func_name].make(arg)
                raise ParseError(f"Unknown function: {func_name}")

            if isinstance(primary, Constant):
                raise ParseError("Cannot call a constant as a function")

            raise ParseError(f"Expected function name, got {primary}")

        return primary

    def parse_primary(self) -> Expr:
        """primary -> NUMBER | IDENT | FUNC '(' expr ')' | '(' expr ')'"""
        tok = self.current

        if tok.kind == "NUMBER":
            self.advance()
            return Constant(float(tok.value))

        if tok.kind == "IDENT":
            self.advance()
            return Symbol(tok.value)

        if tok.kind == "FUNC":
            self.advance()
            self.expect("LPAREN")
            arg = self.parse_expr()
            self.expect("RPAREN")
            return FUNCTION_MAP[tok.value].make(arg)

        if tok.kind == "LPAREN":
            self.advance()
            expr = self.parse_expr()
            self.expect("RPAREN")
            return expr

        raise ParseError(f"Unexpected token: {tok}")


class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""


def parse(source: str) -> Expr:
    """Parse a mathematical expression string into an expression tree.

    Converts a string like ``"x^2 + 2*x + 1"`` into a tree of expression nodes
    (Add, Mul, Pow, Symbol, Constant, etc.) that can be differentiated,
    integrated, evaluated, and manipulated.

    Supported syntax:
        - Numbers: ``42``, ``3.14``
        - Variables: ``x``, ``y``, ``my_var``
        - Operators: ``+``, ``-``, ``*``, ``/``, ``^``
        - Functions: ``sin(x)``, ``cos(x)``, ``tan(x)``, ``exp(x)``, ``ln(x)``,
          ``log(x)``, ``sqrt(x)``, ``abs(x)``, ``asin(x)``, ``acos(x)``,
          ``atan(x)``, ``sec(x)``, ``csc(x)``, ``cot(x)``
        - Parentheses: ``(x + 1) * 2``
        - Unary minus: ``-x``, ``-(x + 1)``

    Args:
        source: A string containing a mathematical expression.

    Returns:
        An ``Expr`` tree representing the parsed expression.

    Raises:
        ParseError: If the input string contains invalid syntax.

    Examples:
        >>> parse("x^2 + sin(x)")
        Add(Pow(Symbol('x'), Constant(2.0)), Sin(Symbol('x')))

        >>> parse("exp(x) / x")
        Div(Exp(Symbol('x')), Symbol('x'))

        >>> parse("1 / (1 + x^2)")
        Div(Constant(1.0), Add(Constant(1.0), Pow(Symbol('x'), Constant(2.0))))
    """
    source = source.strip()
    if not source:
        raise ParseError("Empty expression")
    return Parser(source).parse()
