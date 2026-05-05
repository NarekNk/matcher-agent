from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_bundle(bundle: dict[str, Any], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    model = bundle["model"]
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    joblib.dump(model, target_dir / "model.joblib")
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_bundle(target_dir: Path) -> dict[str, Any]:
    model = joblib.load(target_dir / "model.joblib")
    metadata = json.loads((target_dir / "metadata.json").read_text())
    metadata["model"] = model
    return metadata
