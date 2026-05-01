# Calcus

[![PyPI version](https://img.shields.io/pypi/v/calcus.svg)](https://pypi.org/project/calcus/)
[![Python versions](https://img.shields.io/pypi/pyversions/calcus.svg)](https://pypi.org/project/calcus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/YOUR_USERNAME/calcus/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/calcus/actions/workflows/tests.yml)

A lightweight, dependency-free Python library for symbolic and numerical calculus operations.

## Why Calcus?

- **Zero dependencies** — no SymPy, NumPy, or any external math library required
- **Educational** — transparent implementation of calculus algorithms you can study and modify
- **Simple API** — intuitive functions that do exactly what you expect
- **Lightweight** — no heavy symbolic engine overhead, fast imports, minimal footprint
- **Both symbolic and numerical** — get exact answers when possible, approximations when needed

## Installation

### From PyPI

```bash
pip install calcus
```

### From source

```bash
git clone https://github.com/nalqit/calcus.git
cd calcus
pip install -e .
```

### Development install (with pytest)

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from calcus import parse, differentiate, integrate, limit, pretty

# Parse an expression from a string
expr = parse("x^2 + sin(x)")

# Differentiate
diff_result = differentiate(expr, "x")
print(pretty(diff_result))  # 2 * x + cos(x)

# Integrate
int_result = integrate(expr, "x")
print(pretty(int_result))  # x ^ 3 / 3 - cos(x)

# Limits
lim = limit(parse("sin(x)/x"), "x", 0)
print(lim)  # 1.0
```

## CLI Usage

Calcus ships with a command-line interface:

```bash
# Symbolic differentiation
calcus --diff "x^2 + sin(x)" --var x

# Symbolic integration
calcus --integrate "sin(x)" --var x

# Limits
calcus --limit "sin(x)/x" --var x --point 0

# Taylor series
calcus --taylor "exp(x)" --var x --order 5

# Numerical integration
calcus --num-int "x^2" --var x --lower 0 --upper 1

# LaTeX output
calcus --diff "x^3" --var x --latex

# Interactive REPL
calcus --repl
```

## API Reference

### Parsing

```python
from calcus import parse

expr = parse("x^2 + 2*x + 1")        # Addition and multiplication
expr = parse("sin(x) * cos(x)")      # Functions
expr = parse("exp(x) / x^2")         # Division and powers
expr = parse("sqrt(x) + ln(x)")      # Square root, natural log
expr = parse("atan(x^2)")            # Inverse trig functions
```

**Supported functions:** `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `asin`/`arcsin`, `acos`/`arccos`, `atan`/`arctan`, `exp`, `ln`/`log`, `sqrt`, `abs`

### Symbolic Differentiation

```python
from calcus import parse, differentiate, gradient, hessian

# Basic derivatives
differentiate(parse("x^3"), "x")               # 3 * x ^ 2
differentiate(parse("sin(x)"), "x")            # cos(x)
differentiate(parse("exp(x)"), "x")            # exp(x)
differentiate(parse("ln(x)"), "x")             # 1 / x

# Higher-order derivatives
differentiate(parse("x^4"), "x", order=2)      # 12 * x ^ 2
differentiate(parse("sin(x)"), "x", order=3)   # -cos(x)

# Chain rule
differentiate(parse("sin(x^2)"), "x")          # 2 * x * cos(x ^ 2)

# Product rule
differentiate(parse("x * sin(x)"), "x")        # sin(x) + x * cos(x)

# Quotient rule
differentiate(parse("sin(x) / x"), "x")        # (x * cos(x) - sin(x)) / x ^ 2

# Multivariable
expr = parse("x^2 * y + y^3")
gradient(expr, ["x", "y"])                     # [2 * x * y, x ^ 2 + 3 * y ^ 2]
hessian(expr, ["x", "y"])                      # [[2 * y, 2 * x], [2 * x, 6 * y]]
```

### Symbolic Integration

```python
from calcus import parse, integrate, definite_integral

# Basic antiderivatives
integrate(parse("x^2"), "x")                   # x ^ 3 / 3
integrate(parse("sin(x)"), "x")                # -cos(x)
integrate(parse("cos(x)"), "x")                # sin(x)
integrate(parse("exp(x)"), "x")                # exp(x)
integrate(parse("1/x"), "x")                   # ln(x)

# Definite integrals
definite_integral(parse("x^2"), "x", 0, 1)     # 0.33333...
definite_integral(parse("sin(x)"), "x", 0, 3.14159)  # ~2.0
```

### Limits

```python
from calcus import parse, limit

# Direct substitution
limit(parse("x^2 + 1"), "x", 3)                # 10.0

# 0/0 indeterminate form (L'Hôpital's rule)
limit(parse("sin(x)/x"), "x", 0)               # 1.0

# Limits at infinity
limit(parse("1/x"), "x", "inf")                # 0.0
limit(parse("1/x"), "x", "-inf")               # 0.0
```

### Numerical Differentiation

```python
from calcus import parse, numerical_diff, numerical_diff2

expr = parse("x^2")
numerical_diff(expr, "x", 2)                   # ~4.0
numerical_diff2(expr, "x", 2)                  # ~2.0

# Choose method: 'forward', 'backward', 'central' (default)
numerical_diff(expr, "x", 2, method="central")
```

### Numerical Integration

```python
from calcus import parse, trapezoidal_rule, simpsons_rule, adaptive_quadrature, gaussian_quadrature

expr = parse("x^2")

trapezoidal_rule(expr, "x", 0, 1, n=1000)      # ~0.333
simpsons_rule(expr, "x", 0, 1, n=100)          # ~0.333
adaptive_quadrature(expr, "x", 0, 1)           # ~0.333 (high precision)
gaussian_quadrature(expr, "x", 0, 1, n=3)      # ~0.333
```

### Taylor & Maclaurin Series

```python
from calcus import parse, taylor_series, maclaurin_series, pretty

# Maclaurin series (Taylor at x=0)
series = maclaurin_series(parse("exp(x)"), "x", order=5)
print(pretty(series))  # 1 + x + x^2/2 + x^3/6 + x^4/24

series = maclaurin_series(parse("sin(x)"), "x", order=6)
print(pretty(series))  # x - x^3/6 + x^5/120

# Taylor series at arbitrary point
series = taylor_series(parse("ln(x)"), "x", point=1, order=4)
```

### Vector Calculus

```python
from calcus import parse, gradient, divergence, curl, laplacian

# Gradient of scalar field
f = parse("x^2 + y^2 + z^2")
gradient(f, ["x", "y", "z"])                  # [2*x, 2*y, 2*z]

# Divergence of vector field
F = [parse("x^2"), parse("y^2"), parse("z^2")]
divergence(F, ["x", "y", "z"])                # 2*x + 2*y + 2*z

# Curl of vector field
curl(F, ["x", "y", "z"])                      # [0, 0, 0]

# Laplacian
laplacian(f, ["x", "y", "z"])                 # 6
```

### ODE Solvers

```python
from calcus import ODESolver

# dy/dx = y, y(0) = 1  (solution: y = e^x)
def f(x, y):
    return y

result = ODESolver.runge_kutta_4(f, y0=1, x0=0, x_end=1, h=0.1)
for x, y in result:
    print(f"y({x:.1f}) = {y:.6f}")
```

### Pretty Printing

```python
from calcus import parse, pretty, to_latex

expr = parse("x^2 + 2*x + 1")
print(pretty(expr))     # x ^ 2 + 2 * x + 1
print(to_latex(expr))   # x^{2} + 2 \cdot x + 1
```

### Expression API

```python
from calcus import parse, constant, symbol

# Create expressions programmatically
x = symbol("x")
expr = x ** 2 + 2 * x + 1

# Evaluate with variable bindings
expr = parse("x^2 + 3*x + 2")
expr.evaluate({"x": 2})                     # 12.0

# Substitute variables
expr.substitute("x", constant(3))           # x^2 + 3*x + 2 with x=3
expr.substitute("x", parse("y+1"))          # Replace x with (y+1)

# Get all variables in an expression
parse("x^2 + sin(y)").symbols()             # {"x", "y"}
```

### Complex Example

```python
from calcus import parse, differentiate, integrate, maclaurin_series, pretty

# Differentiate a complex expression
expr = parse("exp(x) * sin(x^2) / (1 + x^2)")
deriv = differentiate(expr, "x")
print(pretty(deriv))

# Expand sin(x) as a Maclaurin series
series = maclaurin_series(parse("sin(x)"), "x", order=8)
print(pretty(series))
# x - x^3 / 6 + x^5 / 120 - x^7 / 5040
```

## Project Structure

```
calcus/
├── calcus/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── core/
│   │   ├── expression.py    # Expression AST nodes
│   │   ├── parser.py        # Recursive descent parser
│   │   ├── simplify.py      # Expression simplification
│   │   └── pretty.py        # ASCII and LaTeX formatting
│   ├── symbolic/
│   │   ├── differentiate.py # Symbolic differentiation
│   │   ├── integrate.py     # Symbolic integration
│   │   └── limits.py        # Limit computation
│   ├── numerical/
│   │   ├── differentiate.py # Numerical differentiation
│   │   └── integrate.py     # Numerical integration
│   └── advanced/
│       ├── series.py        # Taylor/Maclaurin series
│       ├── ode.py           # ODE solvers
│       └── vector.py        # Vector calculus
├── tests/
│   └── test_calculus.py     # All tests
├── pyproject.toml
├── README.md
└── LICENSE
```

## Limitations

- **No implicit multiplication**: Write `2*x` not `2x`
- **Integration is rule-based**: Handles common cases but not a full CAS like SymPy. Complex integrals may raise `NotImplementedError`
- **Simplification is basic**: Does not apply trig identities, log rules, or polynomial factoring
- **Single variable focus**: Multivariable support exists for gradient/Hessian/divergence/curl but not for limits or integration
- **No complex numbers**: All operations are over real numbers

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ -v --cov=calcus
```

## License

MIT
