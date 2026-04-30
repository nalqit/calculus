# Contributing to Calculus

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/calculus.git
cd calculus

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests -v

# Run with coverage
pytest tests -v --cov=calculus

# Run a specific test file or class
pytest tests/test_calculus.py -v
pytest tests/test_calculus.py::TestDifferentiation -v
```

## Code Style

The project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for lint errors
ruff check .

# Auto-fix where possible
ruff check --fix .

# Format code
ruff format .
```

## Pull Request Process

1. **Fork** the repository and create a branch from `main`
2. **Add tests** for any new functionality
3. **Ensure all tests pass**: `pytest tests -v`
4. **Run the linter**: `ruff check .`
5. **Update documentation** if you change the API
6. **Submit a PR** with a clear description of the changes

## What to Contribute

- **Bug fixes**: Fix issues reported in the [issue tracker](https://github.com/YOUR_USERNAME/calculus/issues)
- **New functions**: Add support for more mathematical functions (e.g., `sinh`, `cosh`, `log10`)
- **Better integration**: Expand the symbolic integration engine with more patterns
- **Better simplification**: Add trig identity simplification, polynomial factoring
- **Performance**: Optimize expression evaluation or simplification
- **Documentation**: Improve examples, add tutorials, fix typos
- **Tests**: Add edge cases, improve coverage

## Project Structure

```
calculus/
├── __init__.py          # Public API
├── __main__.py          # CLI entry point
├── core/                # Expression system
│   ├── expression.py    # AST nodes
│   ├── parser.py        # String → AST
│   ├── simplify.py      # Simplification rules
│   └── pretty.py        # Output formatting
├── symbolic/            # Symbolic calculus
│   ├── differentiate.py # Symbolic differentiation
│   ├── integrate.py     # Symbolic integration
│   └── limits.py        # Limit computation
├── numerical/           # Numerical methods
│   ├── differentiate.py # Finite differences
│   └── integrate.py     # Numerical integration
├── advanced/            # Advanced features
│   ├── series.py        # Taylor/Maclaurin series
│   ├── ode.py           # ODE solvers
│   └── vector.py        # Vector calculus
└── tests/               # Unit tests
```

## Naming Conventions

- **Functions**: `snake_case` (e.g., `differentiate`, `numerical_diff`)
- **Classes**: `PascalCase` (e.g., `Constant`, `Symbol`, `ODESolver`)
- **Expression nodes**: Single word or math name (e.g., `Add`, `Mul`, `Sin`)
- **Static factory methods**: `make` (e.g., `Constant.make`, `Pow.make`)

## Adding a New Function

To add a new mathematical function (e.g., `sinh`):

1. **Define the class** in `core/expression.py`:
```python
class Sinh(Function):
    @property
    def name(self) -> str:
        return "sinh"

    @classmethod
    def make(cls, arg: Expr) -> Function:
        if isinstance(arg, Constant):
            from math import sinh
            return Constant(sinh(arg.value))
        return Sinh(arg)

    def _eval(self, x: float) -> float:
        from math import sinh
        return sinh(x)
```

2. **Register it** in `core/parser.py`:
```python
FUNCTION_MAP = {
    ...,
    "sinh": Sinh,
}
```

3. **Add the derivative** in `symbolic/differentiate.py`:
```python
if isinstance(expr, Sinh):
    return Mul.make(Cosh.make(expr.arg), _diff(expr.arg, var))
```

4. **Add formatting** in `core/pretty.py` (both `pretty` and `to_latex`)

5. **Write tests** in `tests/test_calculus.py`

## Questions?

Open an [issue](https://github.com/YOUR_USERNAME/calculus/issues) for questions or discussions.
