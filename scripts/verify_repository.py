#!/usr/bin/env python3
"""Fail on common mistakes before the directory becomes a public Git repo."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = {
    ".py", ".r", ".R", ".sh", ".md", ".yml", ".yaml", ".toml",
    ".txt", ".cff", ".csv", ".lock",
}
PRIVATE_PATH = re.compile(r"/(?:home|media)/[A-Za-z0-9_.-]+/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(errors: list[str]) -> None:
    manifest = ROOT / "data" / "manifest.sha256"
    if not manifest.exists():
        errors.append("missing data/manifest.sha256")
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if not path.exists():
            errors.append(f"manifest target missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")


def main() -> int:
    errors: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in {".git", "external", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if relative.parts[:2] in {("results", "generated"), ("data", "raw"), ("data", "interim")}:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 50 MiB: {relative}")
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore", "external-dependencies.lock"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if PRIVATE_PATH.search(text):
                errors.append(f"private absolute path found: {relative}")

    verify_manifest(errors)
    if errors:
        print("Repository verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Repository verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
