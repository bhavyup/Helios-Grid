from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List


def validate_and_resolve(
    file_path: str,
    allowed_roots: Optional[Iterable[Path]] = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a user-supplied path safely and ensure it lies within allowed roots.

    - Expands user home (`~`) and resolves symlinks without requiring existence.
    - If `must_exist` is True, raises FileNotFoundError when resolved path does not exist.
    - If allowed_roots is provided, the resolved path must be inside one of those roots,
      otherwise a ValueError is raised.

    Returns a resolved Path.
    """
    raw = Path(file_path).expanduser()

    if allowed_roots is None:
        allowed_roots = [Path.cwd().resolve()]

    allowed = [p.resolve() for p in allowed_roots]

    def _is_allowed(p: Path) -> bool:
        try:
            r = p.resolve(strict=False)
        except Exception:
            return False
        for root in allowed:
            try:
                r.relative_to(root)
                return True
            except Exception:
                continue
        return False

    # If absolute path, require it's under an allowed root
    if raw.is_absolute():
        resolved = raw.resolve(strict=False)
        if not _is_allowed(resolved):
            raise ValueError("Absolute paths outside allowed roots are forbidden")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        return resolved

    # For relative paths, try each allowed root as a base
    for root in allowed:
        candidate = (root / raw).resolve(strict=False)
        if candidate.exists():
            if not _is_allowed(candidate):
                continue
            return candidate

    # If path does not exist yet (e.g. output path), return backend-local candidate
    candidate = (allowed[0] / raw).resolve(strict=False)
    if not _is_allowed(candidate):
        raise ValueError("Resolved path is outside allowed roots")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")
    return candidate


__all__ = ["validate_and_resolve"]
