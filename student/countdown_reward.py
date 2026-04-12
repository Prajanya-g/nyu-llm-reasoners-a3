"""Rule-based rewards for Countdown (format + arithmetic + number multiset)."""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
        v = _eval_ast(node.operand)
        return float(_OPS[type(node.op)](v))  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return float(_OPS[type(node.op)](left, right))  # type: ignore[arg-type]
    raise ValueError("unsupported ast")


def _safe_eval_arithmetic(expr: str) -> float:
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expr")
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree)


def _collect_numeric_leaves(node: ast.AST) -> list[float]:
    out: list[float] = []
    if isinstance(node, ast.Expression):
        return _collect_numeric_leaves(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        out.append(float(node.value))
        return out
    if isinstance(node, ast.UnaryOp):
        return _collect_numeric_leaves(node.operand)
    if isinstance(node, ast.BinOp):
        return _collect_numeric_leaves(node.left) + _collect_numeric_leaves(node.right)
    return out


def _extract_answer_block(response: str) -> str | None:
    m = re.search(r"<answer[^>]*>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _final_expr_from_answer_block(block: str) -> str | None:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        if "=" in line:
            rhs = line.split("=")[-1].strip()
            if rhs:
                return rhs
    return lines[-1]


def countdown_reward_fn(response: str, ground_truth: str, *, fast: bool = True) -> dict[str, float]:
    """Score a Countdown rollout.

    ``ground_truth`` is JSON: ``{\"numbers\": [int, ...], \"target\": int}``.

    Args:
        response: Model completion (after prompt).
        ground_truth: JSON string from :func:`json.dumps` over the dict above.
        fast: Unused; kept for a uniform signature with other reward fns.

    Returns:
        ``format_reward``, ``answer_reward``, ``reward`` in ``{0.0, 1.0}``.
    """
    _ = fast
    try:
        spec: dict[str, Any] = json.loads(ground_truth)
        nums = [int(x) for x in spec["numbers"]]
        target = float(spec["target"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    block = _extract_answer_block(response)
    if block is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    expr = _final_expr_from_answer_block(block)
    if expr is None:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    try:
        tree = ast.parse(expr, mode="eval")
        assert isinstance(tree, ast.Expression)
        leaves = _collect_numeric_leaves(tree.body)
        value = _eval_ast(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, AssertionError):
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    want = sorted(float(n) for n in nums)
    got = sorted(leaves)
    if len(got) != len(want) or got != want:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    if abs(value - target) > 1e-4 * max(1.0, abs(target)):
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
