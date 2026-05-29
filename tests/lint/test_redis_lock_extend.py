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

_REDIS_PACKAGE = pathlib.Path("token_throttle/_limiter_backends/_redis")


def _body_contains_extend_locks(body: list[ast.stmt]) -> bool:
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == "_extend_locks":
                return True
    return False


def _is_self_lock_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_lock"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _lock_call_from_context_expr(node: ast.AST) -> ast.Call | None:
    if _is_self_lock_call(node):
        return node  # sync pattern: self._lock(...)
    if isinstance(node, ast.Await) and _is_self_lock_call(node.value):
        return node.value  # async pattern: await self._lock(...)
    return None


def _has_lock_context(source: str, filepath: str) -> bool:
    tree = ast.parse(source, filename=filepath)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncWith, ast.With)):
            continue
        if any(_lock_call_from_context_expr(item.context_expr) for item in node.items):
            return True
    return False


def _redis_files_with_lock_contexts(repo_root: pathlib.Path) -> list[str]:
    redis_root = repo_root / _REDIS_PACKAGE
    files: list[str] = []
    for path in sorted(redis_root.rglob("*.py")):
        relative_path = path.relative_to(repo_root).as_posix()
        if _has_lock_context(path.read_text(encoding="utf-8"), relative_path):
            files.append(relative_path)
    return files


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

            if _lock_call_from_context_expr(ctx) is None:
                continue

            if not _body_contains_extend_locks(node.body):
                violations.append(
                    f"{filepath}:{node.lineno}: "
                    "`_lock` context has no `_extend_locks` call in body"
                )

    return violations


def test_every_redis_lock_context_has_extend_locks() -> None:
    repo_root = pathlib.Path(__file__).parent.parent.parent
    redis_backend_files = _redis_files_with_lock_contexts(repo_root)
    assert redis_backend_files, "No Redis files with self._lock contexts discovered"

    all_violations: list[str] = []
    for relative_path in redis_backend_files:
        path = repo_root / relative_path
        source = path.read_text(encoding="utf-8")
        all_violations.extend(_find_lock_contexts_without_extend(source, relative_path))

    assert not all_violations, (
        "Redis _lock contexts missing _extend_locks — "
        "add _extend_locks(lock_stack) before any write:\n" + "\n".join(all_violations)
    )
