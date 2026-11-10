from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("models/best_telemetry_model.pkl")
DEFAULT_METADATA_PATH = Path("models/telemetry_only_metadata.json")


def save_model(model: Any, path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def save_metadata(metadata: dict, path: Path = DEFAULT_METADATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: Path = DEFAULT_METADATA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
