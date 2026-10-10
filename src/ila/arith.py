"""Safe evaluation, formatting, and manipulation of GSM8K calculator expressions."""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_OP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}


class ArithError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ArithError("exponent too large")
        try:
            return _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as e:
            raise ArithError("division by zero") from e
    raise ArithError(f"unsupported node {type(node).__name__}")


def normalize_expr(expr: str) -> str:
    """Convert GSM8K calculator syntax (commas, %, x) into Python arithmetic."""
    e = expr.strip().replace(",", "").replace("x", "*").replace("^", "**")
    e = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", e)
    return e


def safe_eval(expr: str) -> float:
    try:
        tree = ast.parse(normalize_expr(expr), mode="eval")
    except SyntaxError as e:
        raise ArithError(f"cannot parse {expr!r}") from e
    v = _eval(tree)
    if not math.isfinite(v):
        raise ArithError("non-finite result")
    return v


def parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


def is_int(x: float) -> bool:
    return abs(x - round(x)) < 1e-9


def display_expr(expr: str) -> str:
    """Human-readable rendering with spaced binary operators."""
    e = normalize_expr(expr)
    e = re.sub(r"(?<=[\d)])\s*(\*\*|[+\-*/])\s*", r" \1 ", e)
    return re.sub(r"\s+", " ", e).strip()


@dataclass(frozen=True)
class Binary:
    a: float
    op: str
    b: float


def as_binary(expr: str) -> Binary | None:
    """Return (a, op, b) if expr is a single binary operation on two literals."""
    try:
        tree = ast.parse(normalize_expr(expr), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(tree, ast.BinOp) and type(tree.op) in _OP_SYMBOL):
        return None
    try:
        a, b = _eval(tree.left), _eval(tree.right)
    except ArithError:
        return None
    lit = (ast.Constant, ast.UnaryOp)
    if not (isinstance(tree.left, lit) and isinstance(tree.right, lit)):
        return None
    return Binary(a, _OP_SYMBOL[type(tree.op)], b)


def inverse_check(expr: str, result: float) -> str | None:
    """An inverse-operation identity that re-derives an operand from the result."""
    bn = as_binary(expr)
    if bn is None:
        return None
    a, b, c = fmt(bn.a), fmt(bn.b), fmt(result)
    if bn.op == "*" and bn.b != 0:
        return f"{c} / {b} = {a}"
    if bn.op == "/":
        return f"{c} * {b} = {a}"
    if bn.op == "+":
        return f"{c} - {b} = {a}"
    if bn.op == "-":
        return f"{c} + {b} = {a}"
    return None


_WRONG_OPS = {"*": ["+", "-"], "+": ["*", "-"], "-": ["+", "*"], "/": ["*", "-"], "**": ["*", "+"]}


def wrong_attempts(expr: str, result: float) -> list[tuple[str, float]]:
    """Plausible erroneous computations (operator swap or operand slip) with results != result."""
    out: list[tuple[str, float]] = []
    seen = {round(result, 6)}
    bn = as_binary(expr)
    candidates: list[str] = []
    if bn is not None:
        for op in _WRONG_OPS[bn.op]:
            candidates.append(f"{fmt(bn.a)} {op} {fmt(bn.b)}")
    nums = list(re.finditer(r"\d+(?:\.\d+)?", normalize_expr(expr)))
    if nums:
        last = nums[-1]
        e = normalize_expr(expr)
        for delta in (1, 10):
            bumped = fmt(float(last.group()) + delta)
            candidates.append(display_expr(e[: last.start()] + bumped + e[last.end():]))
    for cand in candidates:
        try:
            v = safe_eval(cand)
        except ArithError:
            continue
        if round(v, 6) in seen or abs(v) > 1e9:
            continue
        seen.add(round(v, 6))
        out.append((display_expr(cand), v))
    return out


def decomposition(expr: str, result: float) -> str | None:
    """A forward (non-revisiting) place-value breakdown of a multi-digit integer operation."""
    bn = as_binary(expr)
    if bn is None or not (is_int(bn.a) and is_int(bn.b) and bn.a >= 0 and bn.b >= 0):
        return None
    a, b = int(round(bn.a)), int(round(bn.b))
    if b < 10 or b % 10 == 0:
        return None
    tens, ones = b - b % 10, b % 10
    if bn.op == "*":
        x, y = a * tens, a * ones
        return f"Breaking this down: {a} * {tens} = {x} and {a} * {ones} = {y}, so {x} + {y} = {fmt(result)}."
    if bn.op == "+":
        x = a + tens
        return f"Breaking this down: {a} + {tens} = {x}, then {x} + {ones} = {fmt(result)}."
    if bn.op == "-" and a >= b:
        x = a - tens
        return f"Breaking this down: {a} - {tens} = {x}, then {x} - {ones} = {fmt(result)}."
    return None
