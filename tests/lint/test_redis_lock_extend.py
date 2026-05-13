"""
D34 regression marker — R5 L27:LK07.

Every `with self._lock(...)` or `async with await self._lock(...)` context in
the Redis backend files must contain at least one `_extend_locks` call in its
body.  This invariant was confirmed clean in the R5 LK07 audit; this test
makes a future omission immediately visible.

KNOWN UNKNOWN: the check walks the AST body recursively, so _extend_locks
buried inside nested try/for/if blocks counts as present. It does NOT verify
that _extend_locks is called *before* the first write — only that it is called
at all within the lock scope. A more granular dataflow check is deferred.
"""

import ast
import pathlib

_REDIS_BACKEND_FILES = [
    "token_throttle/_limiter_backends/_redis/_backend.py",
    "token_throttle/_limiter_backends/_redis/_sync_backend.py",
]


def _body_contains_extend_locks(body: list[ast.stmt]) -> bool:
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == "_extend_locks":
                return True
    return False


def _find_lock_contexts_without_extend(source: str, filepath: str) -> list[str]:
    tree = ast.parse(source, filename=filepath)
    violations: list[str] = []

    for node in ast.walk(tree):
        # Handle both async (`async with await self._lock(...)`) and
        # sync (`with self._lock(...)`) context managers.
        if isinstance(node, (ast.AsyncWith, ast.With)):
            items = node.items
        else:
            continue

        for item in items:
            ctx = item.context_expr

            # async pattern: await self._lock(...)
            is_async_lock = (
                isinstance(ctx, ast.Await)
                and isinstance(ctx.value, ast.Call)
                and isinstance(ctx.value.func, ast.Attribute)
                and ctx.value.func.attr == "_lock"
            )

            # sync pattern: self._lock(...)
            is_sync_lock = (
                isinstance(ctx, ast.Call)
                and isinstance(ctx.func, ast.Attribute)
                and ctx.func.attr == "_lock"
            )

            if not (is_async_lock or is_sync_lock):
                continue

            if not _body_contains_extend_locks(node.body):
                violations.append(
                    f"{filepath}:{node.lineno}: "
                    "`_lock` context has no `_extend_locks` call in body"
                )

    return violations


def test_every_redis_lock_context_has_extend_locks() -> None:
    repo_root = pathlib.Path(__file__).parent.parent.parent

    all_violations: list[str] = []
    for relative_path in _REDIS_BACKEND_FILES:
        path = repo_root / relative_path
        source = path.read_text(encoding="utf-8")
        all_violations.extend(_find_lock_contexts_without_extend(source, relative_path))

    assert not all_violations, (
        "Redis _lock contexts missing _extend_locks — "
        "add _extend_locks(lock_stack) before any write:\n" + "\n".join(all_violations)
    )
