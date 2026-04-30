from calcus import (
    Add,
    Constant,
    Mul,
    ParseError,
    Pow,
    Sin,
    Symbol,
    adaptive_quadrature,
    curl,
    definite_integral,
    differentiate,
    divergence,
    gaussian_quadrature,
    gradient,
    hessian,
    integrate,
    laplacian,
    limit,
    maclaurin_series,
    numerical_diff,
    numerical_diff2,
    parse,
    pretty,
    simpsons_rule,
    simplify,
    taylor_series,
    to_latex,
    trapezoidal_rule,
)
import pytest
from math import exp as math_exp, pi, sin as math_sin


class TestParsing:
    def test_constant(self):
        expr = parse("42")
        assert expr == Constant(42)

    def test_symbol(self):
        expr = parse("x")
        assert expr == Symbol("x")

    def test_addition(self):
        expr = parse("x + 1")
        assert isinstance(expr, Add)

    def test_multiplication(self):
        expr = parse("2 * x")
        assert isinstance(expr, Mul)

    def test_power(self):
        expr = parse("x ^ 2")
        assert isinstance(expr, Pow)

    def test_function(self):
        expr = parse("sin(x)")
        assert isinstance(expr, Sin)

    def test_nested(self):
        expr = parse("x ^ 2 + 2 * x + 1")
        assert expr.symbols() == {"x"}

    def test_implicit_multiplication_not_supported(self):
        with pytest.raises(Exception):
            parse("2x")


class TestDifferentiation:
    def test_constant(self):
        result = differentiate(Constant(5), "x")
        assert result == Constant(0)

    def test_power_rule(self):
        expr = parse("x ^ 2")
        result = differentiate(expr, "x")
        assert pretty(result) == "2 * x"

    def test_sum_rule(self):
        expr = parse("x ^ 3 + x ^ 2")
        result = differentiate(expr, "x")
        assert "x" in pretty(result)

    def test_sin(self):
        expr = parse("sin(x)")
        result = differentiate(expr, "x")
        assert pretty(result) == "cos(x)"

    def test_cos(self):
        expr = parse("cos(x)")
        result = differentiate(expr, "x")
        expected = simplify(result)
        assert "-sin(x)" in pretty(result)

    def test_exp(self):
        expr = parse("exp(x)")
        result = differentiate(expr, "x")
        assert pretty(result) == "exp(x)"

    def test_ln(self):
        expr = parse("ln(x)")
        result = differentiate(expr, "x")
        assert pretty(result) == "1 / x"

    def test_chain_rule(self):
        expr = parse("sin(x ^ 2)")
        result = differentiate(expr, "x")
        assert "x" in pretty(result)

    def test_product_rule(self):
        expr = parse("x * sin(x)")
        result = differentiate(expr, "x")
        assert pretty(result) == "sin(x) + x * cos(x)"

    def test_quotient_rule(self):
        expr = parse("sin(x) / x")
        result = differentiate(expr, "x")
        assert "x" in pretty(result)

    def test_higher_order(self):
        expr = parse("x ^ 4")
        result = differentiate(expr, "x", order=2)
        assert pretty(result) == "12 * x ^ 2"

    def test_partial_derivative(self):
        expr = parse("x ^ 2 * y + 3 * y ^ 2")
        dx = differentiate(expr, "x")
        dy = differentiate(expr, "y")
        assert "x" in pretty(dx)
        assert "y" in pretty(dy)


class TestIntegration:
    def test_constant(self):
        expr = Constant(3)
        result = integrate(expr, "x")
        assert "x" in pretty(result)

    def test_power_rule(self):
        expr = parse("x ^ 2")
        result = integrate(expr, "x")
        assert pretty(result) == "x ^ 3 / 3"

    def test_x(self):
        expr = parse("x")
        result = integrate(expr, "x")
        assert "x ^ 2" in pretty(result)

    def test_sin(self):
        expr = parse("sin(x)")
        result = integrate(expr, "x")
        assert "cos(x)" in pretty(result)

    def test_cos(self):
        expr = parse("cos(x)")
        result = integrate(expr, "x")
        assert pretty(result) == "sin(x)"

    def test_exp(self):
        expr = parse("exp(x)")
        result = integrate(expr, "x")
        assert pretty(result) == "exp(x)"

    def test_1_over_x(self):
        expr = parse("1 / x")
        result = integrate(expr, "x")
        assert "ln(x)" in pretty(result)

    def test_sum(self):
        expr = parse("x ^ 2 + sin(x)")
        result = integrate(expr, "x")
        assert "x ^ 3" in pretty(result)
        assert "cos(x)" in pretty(result)

    def test_definite_integral(self):
        expr = parse("x ^ 2")
        result = definite_integral(expr, "x", 0, 1)
        assert abs(result - 1 / 3) < 1e-10


class TestLimits:
    def test_direct_substitution(self):
        expr = parse("x ^ 2 + 1")
        result = limit(expr, "x", 3)
        assert result == 10

    def test_zero_over_zero(self):
        expr = parse("sin(x) / x")
        result = limit(expr, "x", 0)
        assert abs(result - 1) < 1e-6

    def test_limit_at_infinity(self):
        expr = parse("1 / x")
        result = limit(expr, "x", "inf")
        assert abs(result) < 1e-6


class TestSimplify:
    def test_add_zero(self):
        expr = parse("x + 0")
        result = simplify(expr)
        assert pretty(result) == "x"

    def test_mul_one(self):
        expr = parse("1 * x")
        result = simplify(expr)
        assert pretty(result) == "x"

    def test_mul_zero(self):
        expr = parse("0 * x")
        result = simplify(expr)
        assert pretty(result) == "0"

    def test_power_zero(self):
        expr = parse("x ^ 0")
        result = simplify(expr)
        assert pretty(result) == "1"

    def test_power_one(self):
        expr = parse("x ^ 1")
        result = simplify(expr)
        assert pretty(result) == "x"


class TestPretty:
    def test_simple(self):
        expr = parse("x + 1")
        assert "+" in pretty(expr)

    def test_nested(self):
        expr = parse("x ^ 2 + 2 * x + 1")
        s = pretty(expr)
        assert "^" in s
        assert "+" in s

    def test_latex(self):
        expr = parse("x ^ 2")
        latex = to_latex(expr)
        assert "x^{2}" in latex or "x^2" in latex


class TestNumericalDiff:
    def test_x_squared_at_2(self):
        expr = parse("x ^ 2")
        result = numerical_diff(expr, "x", 2)
        assert abs(result - 4) < 1e-5

    def test_sin_at_0(self):
        expr = parse("sin(x)")
        result = numerical_diff(expr, "x", 0)
        assert abs(result - 1) < 1e-5

    def test_exp_at_0(self):
        expr = parse("exp(x)")
        result = numerical_diff(expr, "x", 0)
        assert abs(result - 1) < 1e-5

    def test_second_derivative(self):
        expr = parse("x ^ 3")
        result = numerical_diff2(expr, "x", 2)
        assert abs(result - 12) < 1e-3


class TestNumericalIntegration:
    def test_trapezoidal_x_squared(self):
        expr = parse("x ^ 2")
        result = trapezoidal_rule(expr, "x", 0, 1, n=10000)
        assert abs(result - 1 / 3) < 1e-4

    def test_simpsons_x_squared(self):
        expr = parse("x ^ 2")
        result = simpsons_rule(expr, "x", 0, 1, n=100)
        assert abs(result - 1 / 3) < 1e-10

    def test_adaptive_sin(self):
        expr = parse("sin(x)")
        result = adaptive_quadrature(expr, "x", 0, pi)
        assert abs(result - 2) < 1e-8

    def test_gaussian_x_squared(self):
        expr = parse("x ^ 2")
        result = gaussian_quadrature(expr, "x", 0, 1, n=3)
        assert abs(result - 1 / 3) < 1e-10


class TestSeries:
    def test_exp_maclaurin(self):
        expr = parse("exp(x)")
        series = maclaurin_series(expr, "x", order=5)
        s = pretty(series)
        assert "x ^ 4" in s or "x" in s

    def test_sin_maclaurin(self):
        expr = parse("sin(x)")
        series = maclaurin_series(expr, "x", order=6)
        s = pretty(series)
        assert "x" in s

    def test_cos_maclaurin(self):
        expr = parse("cos(x)")
        series = maclaurin_series(expr, "x", order=6)
        s = pretty(series)
        assert "x" in s


class TestVectorCalculus:
    def test_gradient(self):
        expr = parse("x ^ 2 + y ^ 2")
        grad = gradient(expr, ["x", "y"])
        assert pretty(grad[0]) == "2 * x"
        assert pretty(grad[1]) == "2 * y"

    def test_divergence(self):
        F = [parse("x ^ 2"), parse("y ^ 2")]
        div = divergence(F, ["x", "y"])
        assert pretty(div) == "2 * x + 2 * y"

    def test_laplacian(self):
        expr = parse("x ^ 2 + y ^ 2")
        lap = laplacian(expr, ["x", "y"])
        assert pretty(lap) == "4"


class TestEvaluation:
    def test_constant(self):
        expr = parse("42")
        assert expr.evaluate({}) == 42

    def test_symbol(self):
        expr = parse("x")
        assert expr.evaluate({"x": 5}) == 5

    def test_complex(self):
        expr = parse("x ^ 2 + 2 * x + 1")
        assert expr.evaluate({"x": 3}) == 16

    def test_function(self):
        expr = parse("sin(x)")
        result = expr.evaluate({"x": 0})
        assert abs(result) < 1e-10


class TestSubstitution:
    def test_symbol_sub(self):
        expr = parse("x ^ 2 + 1")
        result = expr.substitute("x", Constant(3))
        assert result.evaluate({}) == 10

    def test_nested_sub(self):
        expr = parse("sin(x)")
        result = expr.substitute("x", Constant(0))
        assert abs(result.evaluate({})) < 1e-10


class TestSymbols:
    def test_single(self):
        expr = parse("x ^ 2")
        assert expr.symbols() == {"x"}

    def test_multiple(self):
        expr = parse("x + y")
        assert expr.symbols() == {"x", "y"}

    def test_nested(self):
        expr = parse("sin(x) + y ^ 2")
        assert expr.symbols() == {"x", "y"}


class TestEdgeCases:
    def test_empty_string(self):
        with pytest.raises(ParseError):
            parse("")

    def test_whitespace_only(self):
        with pytest.raises(ParseError):
            parse("   ")

    def test_unknown_function(self):
        with pytest.raises(ParseError):
            parse("unknown_func(x)")

    def test_unmatched_paren(self):
        with pytest.raises(ParseError):
            parse("(x + 1")

    def test_double_negation(self):
        expr = parse("--x")
        result = simplify(expr)
        assert pretty(result) == "x"

    def test_negation_of_constant(self):
        expr = parse("-5")
        assert expr.evaluate({}) == -5

    def test_nested_functions(self):
        expr = parse("sin(cos(x))")
        result = differentiate(expr, "x")
        assert "cos" in pretty(result)
        assert "sin" in pretty(result)

    def test_zero_derivative(self):
        result = differentiate(Constant(42), "x")
        assert result == Constant(0)

    def test_derivative_with_respect_to_other_var(self):
        expr = parse("x ^ 2")
        result = differentiate(expr, "y")
        assert result == Constant(0)

    def test_integral_of_constant_zero(self):
        result = integrate(Constant(0), "x")
        assert result == Constant(0)

    def test_third_derivative(self):
        expr = parse("x ^ 3 + 2 * x ^ 2 + x")
        result = differentiate(expr, "x", order=3)
        s = pretty(result)
        assert "6" in s

    def test_limit_does_not_exist(self):
        with pytest.raises(ValueError):
            limit(parse("1/x"), "x", 0)

    def test_numerical_diff_methods(self):
        expr = parse("x ^ 3")
        forward = numerical_diff(expr, "x", 2, method="forward")
        backward = numerical_diff(expr, "x", 2, method="backward")
        central = numerical_diff(expr, "x", 2, method="central")
        assert abs(forward - 12) < 1e-4
        assert abs(backward - 12) < 1e-4
        assert abs(central - 12) < 1e-6

    def test_numerical_diff_invalid_method(self):
        expr = parse("x ^ 2")
        with pytest.raises(ValueError):
            numerical_diff(expr, "x", 1, method="invalid")

    def test_simplify_complex_expression(self):
        expr = parse("x * 0 + 1 * x + 0")
        result = simplify(expr)
        assert pretty(result) == "x"

    def test_division_by_constant_zero(self):
        with pytest.raises(ZeroDivisionError):
            parse("1 / 0")

    def test_evaluation_missing_variable(self):
        expr = parse("x + y")
        with pytest.raises(ValueError):
            expr.evaluate({"x": 1})

    def test_multiple_constants_in_mul(self):
        expr = parse("2 * 3 * x")
        result = simplify(expr)
        assert pretty(result) == "6 * x"

    def test_chain_of_functions(self):
        expr = parse("exp(sin(x))")
        result = differentiate(expr, "x")
        s = pretty(result)
        assert "cos(x)" in s
        assert "exp(sin(x))" in s

    def test_definite_integral_zero_to_pi(self):
        result = definite_integral(parse("sin(x)"), "x", 0, pi)
        assert abs(result - 2) < 1e-8

    def test_limit_infinity_polynomial(self):
        result = limit(parse("1 / (x ^ 2 + 1)"), "x", "inf")
        assert abs(result) < 1e-6

    def test_taylor_sin_odd_terms_only(self):
        series = maclaurin_series(parse("sin(x)"), "x", order=7)
        s = pretty(series)
        assert "x ^ 2" not in s
        assert "x ^ 4" not in s
        assert "x ^ 6" not in s

    def test_hessian_symmetric(self):
        expr = parse("x ^ 2 * y + x * y ^ 2")
        h = hessian(expr, ["x", "y"])
        assert pretty(h[0][1]) == pretty(h[1][0])
