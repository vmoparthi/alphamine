"""Alpha objects + a SAFE evaluator for DSL expressions.

We never call Python's eval() on arbitrary strings. Instead we parse the expression
into an AST and walk it, allowing only:
  - field names (open/high/low/close/...)            -> panel DataFrames
  - numeric constants
  - whitelisted operator calls (dsl.OPERATORS)
  - +, -, *, /, **, and unary minus
Anything else (attributes, imports, names we don't know, lambdas...) raises AlphaError.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import dsl


class AlphaError(ValueError):
    """Raised when an expression is malformed or references unknown symbols."""


_BINOPS = {
    ast.Add: dsl.add, ast.Sub: dsl.sub, ast.Mult: dsl.mul,
    ast.Div: dsl.div, ast.Pow: dsl.pow_,
}


class _Evaluator(ast.NodeVisitor):
    def __init__(self, env: Dict[str, pd.DataFrame]):
        self.env = env

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise AlphaError(f"operator {type(node.op).__name__} not allowed")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise AlphaError("only unary +/- allowed")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise AlphaError("only direct operator calls allowed")
        name = node.func.id
        if name not in dsl.OPERATORS:
            raise AlphaError(f"unknown operator: {name}")
        if node.keywords:
            raise AlphaError("keyword arguments not allowed")
        args = [self.visit(a) for a in node.args]
        return dsl.OPERATORS[name](*args)

    def visit_Name(self, node):
        if node.id not in self.env:
            raise AlphaError(f"unknown field/name: {node.id}")
        return self.env[node.id]

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise AlphaError(f"only numeric constants allowed, got {node.value!r}")

    # python<3.8 compatibility
    def visit_Num(self, node):  # pragma: no cover
        return node.n

    def generic_visit(self, node):
        raise AlphaError(f"syntax element not allowed: {type(node).__name__}")


@dataclass
class Alpha:
    expr: str
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, panel) -> pd.DataFrame:
        """Return the raw signal DataFrame (dates x tickers) for this expression."""
        env = {f: panel.fields[f] for f in dsl.FIELDS if f in panel.fields}
        try:
            tree = ast.parse(self.expr, mode="eval")
        except SyntaxError as e:
            raise AlphaError(f"could not parse: {e}") from e
        signal = _Evaluator(env).visit(tree)
        if not isinstance(signal, pd.DataFrame):
            raise AlphaError("expression did not produce a panel (got a scalar?)")
        return signal.replace([np.inf, -np.inf], np.nan)

    def __str__(self):
        return self.expr


def validate(expr: str, panel) -> Alpha:
    """Parse + evaluate on a small slice to confirm the expression is well-formed."""
    a = Alpha(expr=expr)
    sig = a.evaluate(panel)
    if sig.dropna(how="all").shape[0] == 0:
        raise AlphaError("expression produced all-NaN signal")
    return a
