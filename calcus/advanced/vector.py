from __future__ import annotations

from ..core.expression import Add, Expr, Mul, Symbol
from ..core.simplify import simplify
from ..symbolic.differentiate import differentiate


def gradient(scalar_field: Expr, variables: list[str]) -> list[Expr]:
    """Compute the gradient of a scalar field.

    The gradient is the vector of partial derivatives:
    ``∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]``.

    Args:
        scalar_field: The scalar field expression.
        variables: List of variable names.

    Returns:
        List of partial derivative expressions.

    Examples:
        >>> from calcus import parse, gradient, pretty
        >>> grad = gradient(parse("x^2 + y^2 + z^2"), ["x", "y", "z"])
        >>> [pretty(g) for g in grad]
        ['2 * x', '2 * y', '2 * z']
    """
    return [simplify(differentiate(scalar_field, v)) for v in variables]


def divergence(vector_field: list[Expr], variables: list[str]) -> Expr:
    """Compute the divergence of a vector field.

    ``div(F) = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z``

    The vector field and variables must have the same length.

    Args:
        vector_field: List of component expressions [F₁, F₂, ...].
        variables: List of variable names [x, y, ...].

    Returns:
        The divergence expression.

    Raises:
        ValueError: If the vector field length doesn't match the number of variables.

    Examples:
        >>> from calcus import parse, divergence, pretty
        >>> div = divergence([parse("x^2"), parse("y^2")], ["x", "y"])
        >>> pretty(div)
        '2 * x + 2 * y'
    """
    if len(vector_field) != len(variables):
        raise ValueError(
            f"Vector field has {len(vector_field)} components but {len(variables)} variables given"
        )

    terms = [
        differentiate(comp, var)
        for comp, var in zip(vector_field, variables)
    ]
    return simplify(Add.make(*terms))


def curl(
    vector_field: list[Expr], variables: list[str]
) -> list[Expr]:
    """Compute the curl of a 3D vector field.

    ``curl(F) = (∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y)``

    Only defined for 3-dimensional vector fields.

    Args:
        vector_field: List of 3 component expressions [F₁, F₂, F₃].
        variables: List of 3 variable names [x, y, z].

    Returns:
        List of 3 expressions representing the curl vector.

    Raises:
        ValueError: If the vector field does not have exactly 3 components.

    Examples:
        >>> from calcus import parse, curl, pretty
        >>> curl([parse("y"), parse("-x"), parse("0")], ["x", "y", "z"])
        [0, 0, -2]
    """
    if len(vector_field) != 3 or len(variables) != 3:
        raise ValueError("Curl requires exactly 3 components and 3 variables")

    F1, F2, F3 = vector_field
    x, y, z = variables

    curl_x = differentiate(F3, y) - differentiate(F2, z)
    curl_y = differentiate(F1, z) - differentiate(F3, x)
    curl_z = differentiate(F2, x) - differentiate(F1, y)

    return [simplify(curl_x), simplify(curl_y), simplify(curl_z)]


def laplacian(scalar_field: Expr, variables: list[str]) -> Expr:
    """Compute the Laplacian of a scalar field.

    ``∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²``

    The sum of second partial derivatives with respect to each variable.

    Args:
        scalar_field: The scalar field expression.
        variables: List of variable names.

    Returns:
        The Laplacian expression.

    Examples:
        >>> from calcus import parse, laplacian, pretty
        >>> laplacian(parse("x^2 + y^2"), ["x", "y"])
        4
    """
    terms = [
        differentiate(scalar_field, v, order=2) for v in variables
    ]
    return simplify(Add.make(*terms))
