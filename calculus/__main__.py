"""Command-line interface for the calculus library.

Usage:
    python -m calculus [OPTIONS]

Options:
    --diff EXPR           Compute the symbolic derivative of EXPR
    --integrate EXPR      Compute the symbolic integral of EXPR
    --limit EXPR          Compute the limit of EXPR
    --taylor EXPR         Compute the Taylor series of EXPR
    --num-diff EXPR       Compute numerical derivative at a point
    --num-int EXPR        Compute numerical definite integral
    --var VAR             Variable name (default: x)
    --point POINT         Point for differentiation or limits (default: 0)
    --lower LOWER         Lower bound for integration (default: 0)
    --upper UPPER         Upper bound for integration (default: 1)
    --order ORDER         Order for derivatives or series (default: 1)
    --latex               Output in LaTeX format
    --repl                Start interactive REPL mode
"""

from __future__ import annotations

import argparse
import sys

from calculus.core.parser import ParseError
from calculus.core.pretty import pretty, to_latex
from calculus.symbolic.differentiate import differentiate
from calculus.symbolic.integrate import integrate
from calculus.symbolic.limits import limit


def _parse_expr(source: str):
    """Parse an expression string, printing a helpful error on failure."""
    from calculus import parse

    try:
        return parse(source)
    except ParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"  Expression: {source}", file=sys.stderr)
        sys.exit(1)


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the --diff command."""
    expr = _parse_expr(args.diff)
    result = differentiate(expr, args.var, order=args.order)
    output = to_latex(result) if args.latex else pretty(result)
    if args.latex:
        print(f"$${output}$$")
    else:
        print(output)


def cmd_integrate(args: argparse.Namespace) -> None:
    """Handle the --integrate command."""
    expr = _parse_expr(args.integrate)
    result = integrate(expr, args.var)
    output = to_latex(result) if args.latex else pretty(result)
    if args.latex:
        print(f"$${output} + C$$")
    else:
        print(f"{output} + C")


def cmd_limit(args: argparse.Namespace) -> None:
    """Handle the --limit command."""
    expr = _parse_expr(args.limit)
    try:
        result = limit(expr, args.var, args.point)
        print(result)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_taylor(args: argparse.Namespace) -> None:
    """Handle the --taylor command."""
    from calculus import maclaurin_series, taylor_series

    expr = _parse_expr(args.taylor)
    if args.point == 0:
        result = maclaurin_series(expr, args.var, order=args.order)
    else:
        result = taylor_series(expr, args.var, point=args.point, order=args.order)
    output = to_latex(result) if args.latex else pretty(result)
    print(output)


def cmd_num_diff(args: argparse.Namespace) -> None:
    """Handle the --num-diff command."""
    from calculus import numerical_diff

    expr = _parse_expr(args.num_diff)
    result = numerical_diff(expr, args.var, args.point)
    print(result)


def cmd_num_int(args: argparse.Namespace) -> None:
    """Handle the --num-int command."""
    from calculus import adaptive_quadrature

    expr = _parse_expr(args.num_int)
    result = adaptive_quadrature(expr, args.var, args.lower, args.upper)
    print(result)


def run_repl() -> None:
    """Run an interactive REPL for calculus operations."""
    from calculus import parse as calc_parse
    from calculus import (
        adaptive_quadrature,
        differentiate,
        integrate,
        limit,
        maclaurin_series,
        numerical_diff,
    )

    print("Calculus REPL")
    print("Commands:")
    print("  diff  <expr> [var] [order]  - Differentiate")
    print("  int   <expr> [var]          - Integrate")
    print("  limit <expr> <var> <point>  - Compute limit")
    print("  taylor <expr> [var] [order] - Taylor/Maclaurin series")
    print("  numdiff <expr> <var> <point> - Numerical derivative")
    print("  numint <expr> <var> <lo> <hi> - Numerical integral")
    print("  eval  <expr> var=value       - Evaluate expression")
    print("  latex <expr>                - Show LaTeX output")
    print("  quit                        - Exit")
    print()

    while True:
        try:
            line = input("calculus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd = parts[0].lower()
        rest = parts[1:]

        if cmd == "diff":
            if len(rest) < 1:
                print("Usage: diff <expr> [var] [order]")
                continue
            expr = calc_parse(" ".join(rest[:1]))
            var = rest[1] if len(rest) > 1 else "x"
            order = int(rest[2]) if len(rest) > 2 else 1
            result = differentiate(expr, var, order=order)
            print(f"  {pretty(result)}")

        elif cmd == "int":
            if len(rest) < 1:
                print("Usage: int <expr> [var]")
                continue
            expr = calc_parse(" ".join(rest[:1]))
            var = rest[1] if len(rest) > 1 else "x"
            result = integrate(expr, var)
            print(f"  {pretty(result)} + C")

        elif cmd == "limit":
            if len(rest) < 3:
                print("Usage: limit <expr> <var> <point>")
                continue
            expr = calc_parse(" ".join(rest[:1]))
            try:
                result = limit(expr, rest[1], float(rest[2]))
                print(f"  {result}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif cmd == "taylor":
            if len(rest) < 1:
                print("Usage: taylor <expr> [var] [order]")
                continue
            expr = calc_parse(" ".join(rest[:1]))
            var = rest[1] if len(rest) > 1 else "x"
            order = int(rest[2]) if len(rest) > 2 else 5
            result = maclaurin_series(expr, var, order=order)
            print(f"  {pretty(result)}")

        elif cmd == "numdiff":
            if len(rest) < 3:
                print("Usage: numdiff <expr> <var> <point>")
                continue
            expr = calc_parse(" ".join(rest[:1]))
            result = numerical_diff(expr, rest[1], float(rest[2]))
            print(f"  {result}")

        elif cmd == "numint":
            if len(rest) < 4:
                print("Usage: numint <expr> <var> <lower> <upper>")
                continue
            expr = calc_parse(" ".join(rest[:1]))
            result = adaptive_quadrature(expr, rest[1], float(rest[2]), float(rest[3]))
            print(f"  {result}")

        elif cmd == "eval":
            if len(rest) < 1 or "=" not in line.split(None, 1)[1]:
                print("Usage: eval <expr> var=value")
                continue
            expr = calc_parse(rest[0])
            var, val = rest[1].split("=")
            result = expr.evaluate({var: float(val)})
            print(f"  {result}")

        elif cmd == "latex":
            if len(rest) < 1:
                print("Usage: latex <expr>")
                continue
            expr = calc_parse(" ".join(rest))
            print(f"  $$ {to_latex(expr)} $$")

        else:
            print(f"Unknown command: {cmd}")
            print("Type 'quit' to exit.")


def main() -> None:
    """Entry point for the calculus CLI."""
    parser = argparse.ArgumentParser(
        prog="calculus",
        description="A from-scratch Python library for symbolic and numerical calculus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m calculus --diff "x^2 + sin(x)" --var x
  python -m calculus --integrate "cos(x)" --var x
  python -m calculus --limit "sin(x)/x" --var x --point 0
  python -m calculus --taylor "exp(x)" --var x --order 5
  python -m calculus --num-diff "x^2" --var x --point 2
  python -m calculus --num-int "x^2" --var x --lower 0 --upper 1
  python -m calculus --diff "x^3" --var x --latex
  python -m calculus --repl
""",
    )

    parser.add_argument("--diff", help="Expression to differentiate")
    parser.add_argument("--integrate", help="Expression to integrate")
    parser.add_argument("--limit", help="Expression to compute limit of")
    parser.add_argument("--taylor", help="Expression to expand as Taylor series")
    parser.add_argument("--num-diff", help="Expression for numerical derivative")
    parser.add_argument("--num-int", help="Expression for numerical integration")
    parser.add_argument("--var", default="x", help="Variable name (default: x)")
    parser.add_argument("--point", type=float, default=0, help="Point for differentiation/limits")
    parser.add_argument("--lower", type=float, default=0, help="Lower bound for integration")
    parser.add_argument("--upper", type=float, default=1, help="Upper bound for integration")
    parser.add_argument("--order", type=int, default=1, help="Order for derivatives or series")
    parser.add_argument("--latex", action="store_true", help="Output in LaTeX format")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL")

    args = parser.parse_args()

    if args.repl:
        run_repl()
        return

    if args.diff:
        cmd_diff(args)
    elif args.integrate:
        cmd_integrate(args)
    elif args.limit:
        cmd_limit(args)
    elif args.taylor:
        cmd_taylor(args)
    elif args.num_diff:
        cmd_num_diff(args)
    elif args.num_int:
        cmd_num_int(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
