from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

from src.api.schemas import SceneRequest


_T = TypeVar("_T")


logger = logging.getLogger(__name__)


def _model_to_plain_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # pydantic v2
    if hasattr(model, "dict"):
        return model.dict()  # pydantic v1
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def _validate_scene(data: Any) -> SceneRequest:
    if hasattr(SceneRequest, "model_validate"):
        return SceneRequest.model_validate(data)  # pydantic v2
    if hasattr(SceneRequest, "parse_obj"):
        return SceneRequest.parse_obj(data)  # pydantic v1
    return SceneRequest(**data)


@dataclass(frozen=True)
class SceneStore:
    path: Path

    @staticmethod
    def default() -> "SceneStore":
        env_path = os.getenv("INFINITE_CANDLE_SCENE_PATH")
        if env_path:
            return SceneStore(Path(env_path))

        base_dir = Path(__file__).resolve().parents[1]
        return SceneStore(base_dir / "scene.json")

    def save(self, scene: SceneRequest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        payload = _model_to_plain_dict(scene)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def load(self) -> Optional[SceneRequest]:
        if not self.path.exists():
            return None

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return _validate_scene(data)
        except Exception:
            logger.exception("Failed to load persisted scene from %s", self.path)
            return None
