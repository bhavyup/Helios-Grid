from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.infrastructure.path_utils import validate_and_resolve


def test_resolve_with_allowed_root(tmp_path):
    allowed = [tmp_path]
    # create a file under root
    f = tmp_path / "subdir" / "file.txt"
    f.parent.mkdir(parents=True)
    f.write_text("hello")

    resolved = validate_and_resolve("subdir/file.txt", allowed_roots=allowed, must_exist=True)
    assert resolved.exists()
    assert resolved == f.resolve()


def test_absolute_outside_allowed_raises(tmp_path):
    allowed = [tmp_path]
    other = Path(tmp_path.parent) / "outside_file.txt"
    other.write_text("x")
    with pytest.raises(ValueError):
        validate_and_resolve(str(other), allowed_roots=allowed)


def test_must_exist_true_raises_for_missing(tmp_path):
    allowed = [tmp_path]
    with pytest.raises(FileNotFoundError):
        validate_and_resolve("missing.txt", allowed_roots=allowed, must_exist=True)
