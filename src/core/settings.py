from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG: Dict[str, Any] | None = None
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path = Path(config_path) if config_path else ROOT_DIR / "config" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config() -> Dict[str, Any]:
    return load_config()


def cfg(*keys: str, default: Any = None) -> Any:
    node = get_config()
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
