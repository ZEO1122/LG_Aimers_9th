"""Check the portfolio's tracked and non-ignored files without deleting anything."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath


EXACT_FILES = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "Preprocess/README.md",
    "Preprocess/requirements.txt",
    "Preprocess/run.py",
    "Preprocess/tests/test_preprocessing_contract.py",
    "tools/check_publication.py",
    "tests/test_publication_guard.py",
    ".github/workflows/portfolio-ci.yml",
}
SECRET_PATTERNS = (
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
)


def candidate_paths(root: Path) -> list[str]:
    """Include tracked files even when a later ignore rule matches them."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(set(result.stdout.decode("utf-8").rstrip("\0").split("\0")) - {""})


def check_paths(root: Path, paths: list[str]) -> list[str]:
    errors = []
    for name in paths:
        path = PurePosixPath(name)
        allowed = name in EXACT_FILES or (
            path.parent == PurePosixPath("Preprocess/common") and path.suffix == ".py"
        ) or (path.parent == PurePosixPath("docs") and path.suffix == ".md")
        if not allowed:
            errors.append(f"{name}: outside the reviewed source/documentation scope")
            continue
        local = root / name
        if local.is_symlink() or not local.is_file():
            errors.append(f"{name}: must be a regular file")
            continue
        if local.stat().st_size > 256_000:
            errors.append(f"{name}: unusually large source/documentation file")
            continue
        try:
            content = local.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: binary or non-UTF-8 content")
            continue
        if "\0" in content:
            errors.append(f"{name}: binary content")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"{name}: possible credential; value withheld")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = candidate_paths(root)
    errors = check_paths(root, paths)
    if errors:
        print("Publication check failed:\n" + "\n".join(errors))
        return 1
    print(f"Publication check passed: {len(paths)} source/documentation files.")
    print("Ignored local files are preserved. This check does not certify sharing rights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
