"""Split the project into two source bundles along a natural division of
labor, for a two-person team to commit their respective parts:

  dist/part1-backend-ml.zip      backend / ML / data pipeline / tests
  dist/part2-frontend-docs.zip   frontend / docs / notebooks / deployment

Generated/heavy/ignored paths (.venv, node_modules, __pycache__, data,
artifacts, build output) are excluded from both.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import PROJECT_ROOT, get_logger  # noqa: E402

logger = get_logger("split_for_git")

PREFIX = "investment-intelligence"
EXCLUDE_PARTS = {"__pycache__", ".venv", "node_modules", ".next", ".git",
                 "dist", "artifacts"}
EXCLUDE_SUFFIX = {".pyc", ".joblib", ".parquet"}

PART1_DIRS = ["backend", "ml", "scripts", "tests"]
PART1_FILES = ["requirements.txt", "requirements-deep.txt", "pytest.ini",
               ".gitignore", ".env.example", "docker/Dockerfile.backend"]

PART2_DIRS = ["frontend", "docs", "notebooks"]
PART2_FILES = ["README.md", "PROJECT_STATUS.md", "docker-compose.yml",
               "docker/Dockerfile.frontend", ".dockerignore"]


def _keep(path: Path) -> bool:
    return not (set(path.parts) & EXCLUDE_PARTS) and path.suffix not in EXCLUDE_SUFFIX


def _build(zip_name: str, dirs: list[str], files: list[str]) -> None:
    out = PROJECT_ROOT / "dist" / zip_name
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in dirs:
            for p in sorted((PROJECT_ROOT / d).rglob("*")):
                if p.is_file() and _keep(p):
                    zf.write(p, f"{PREFIX}/{p.relative_to(PROJECT_ROOT)}")
                    n += 1
        for f in files:
            p = PROJECT_ROOT / f
            if p.exists():
                zf.write(p, f"{PREFIX}/{f}")
                n += 1
    logger.info("wrote %s (%d files, %.1f KB)", out, n, out.stat().st_size / 1024)


def main() -> int:
    _build("part1-backend-ml.zip", PART1_DIRS, PART1_FILES)
    _build("part2-frontend-docs.zip", PART2_DIRS, PART2_FILES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
