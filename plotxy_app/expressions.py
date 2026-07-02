"""Safe math-expression evaluator over data series (Qt-free).

Expressions are parsed with ast and validated against a strict node
whitelist — no eval(). Series are referenced by bare identifiers
(P1 + P2) or by string literals for names that are not valid Python
identifiers ("der(v)" * 2). New functions are added by registering
them in FUNCTIONS; the validator picks them up automatically.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable

import numpy as np


class ExpressionError(Exception):
    """Validation/evaluation error with a user-facing pt-BR message."""


FUNCTIONS: dict[str, Callable] = {
    "abs": np.abs,
    # futuras: "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "log": np.log, ...
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _parse(expr: str) -> ast.Expression:
    if not expr or not expr.strip():
        raise ExpressionError("Informe uma expressão.")
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Expressão inválida: {e.msg}") from e


def _validate_and_collect(node: ast.AST, names: set[str]) -> None:
    """Walk the tree, rejecting anything outside the whitelist and
    collecting series references into `names`."""
    if isinstance(node, ast.Expression):
        _validate_and_collect(node.body, names)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ExpressionError(
                f"Operador não suportado: {type(node.op).__name__}")
        _validate_and_collect(node.left, names)
        _validate_and_collect(node.right, names)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise ExpressionError(
                f"Operador não suportado: {type(node.op).__name__}")
        _validate_and_collect(node.operand, names)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            fname = (node.func.id if isinstance(node.func, ast.Name)
                     else ast.dump(node.func))
            available = ", ".join(sorted(FUNCTIONS))
            raise ExpressionError(
                f"Função desconhecida: {fname}. Disponíveis: {available}")
        if node.keywords:
            raise ExpressionError(
                "Argumentos nomeados não são suportados em funções.")
        for arg in node.args:
            _validate_and_collect(arg, names)
    elif isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            pass  # numeric literal
        elif isinstance(node.value, str):
            names.add(node.value)  # quoted series reference
        else:
            raise ExpressionError(
                f"Constante não suportada: {node.value!r}")
    else:
        raise ExpressionError(
            f"Operação não suportada: {type(node).__name__}")


def collect_series_names(expr: str) -> set[str]:
    """Parse + validate, returning the set of referenced series names."""
    tree = _parse(expr)
    names: set[str] = set()
    _validate_and_collect(tree, names)
    return names


def _eval_node(node: ast.AST, series: dict[str, np.ndarray]):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, series)
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](
            _eval_node(node.left, series), _eval_node(node.right, series))
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, series))
    if isinstance(node, ast.Call):
        args = [_eval_node(a, series) for a in node.args]
        return FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Name):
        return series[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return series[node.value]
        return float(node.value)
    raise AssertionError(f"nó não validado: {node!r}")  # unreachable


def evaluate(expr: str, resolver: Callable[[str], np.ndarray],
             ) -> tuple[np.ndarray, set[str], bool]:
    """Evaluate `expr`, resolving series names through `resolver`.

    Returns (values, used_names, truncated). Arrays of different
    lengths are paired index-wise and truncated to the shortest one.
    Division by zero / overflow produce inf/nan, which downstream code
    renders as gaps.
    """
    tree = _parse(expr)
    names: set[str] = set()
    _validate_and_collect(tree, names)
    if not names:
        raise ExpressionError(
            "A expressão deve referenciar pelo menos uma série.")

    arrays: dict[str, np.ndarray] = {}
    for name in names:
        try:
            arrays[name] = np.asarray(resolver(name), dtype=np.float64)
        except KeyError:
            raise ExpressionError(f'Série não encontrada: "{name}"') from None

    min_len = min(len(a) for a in arrays.values())
    truncated = any(len(a) > min_len for a in arrays.values())
    series = {n: a[:min_len] for n, a in arrays.items()}

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = _eval_node(tree, series)

    values = np.asarray(result, dtype=np.float64)
    if values.ndim != 1:
        raise ExpressionError("A expressão não resultou em uma série.")
    return values, names, truncated
