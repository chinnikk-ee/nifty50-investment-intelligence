"""Package the project into dist/investment-intelligence-kaggle.zip for upload
to Kaggle as a code dataset (see docs/KAGGLE.md and notebooks/kaggle_train.ipynb).

Includes only what the training pipeline needs: backend/, ml/, scripts/,
requirements files and configs — no venv, node_modules, data or artifacts.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import PROJECT_ROOT, get_logger  # noqa: E402

logger = get_logger("package_for_kaggle")

INCLUDE_DIRS = ["backend", "ml", "scripts", "tests"]
INCLUDE_FILES = ["requirements.txt", "requirements-deep.txt", "pytest.ini",
                 "README.md", ".env.example"]
EXCLUDE_PARTS = {"__pycache__", "artifacts", ".pytest_cache"}


def main() -> int:
    dist = PROJECT_ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / "investment-intelligence-kaggle.zip"
    prefix = "investment-intelligence"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in INCLUDE_DIRS:
            for path in sorted((PROJECT_ROOT / d).rglob("*")):
                if path.is_dir() or set(path.parts) & EXCLUDE_PARTS:
                    continue
                if path.suffix in (".pyc", ".joblib"):
                    continue
                zf.write(path, f"{prefix}/{path.relative_to(PROJECT_ROOT)}")
        for f in INCLUDE_FILES:
            p = PROJECT_ROOT / f
            if p.exists():
                zf.write(p, f"{prefix}/{f}")
        # Empty data dirs so the notebook can stage CSVs without mkdir logic.
        for keep in ("data/raw/.gitkeep", "data/processed/.gitkeep"):
            zf.writestr(f"{prefix}/{keep}", "")

    logger.info("wrote %s (%.1f KB)", out, out.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
