# lights-control/src/plugins/registry.py
from __future__ import annotations
import importlib
import inspect
import logging
import pkgutil
from typing import Type

from src.plugins.base import AnimationPlugin

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[AnimationPlugin]] = {}
_DISCOVERED = False


def register_plugin(cls: Type[AnimationPlugin]) -> Type[AnimationPlugin]:
    """Class decorator or direct call to register an AnimationPlugin subclass."""
    if not cls.plugin_id:
        raise ValueError(f"Plugin class {cls.__name__} must define a non-empty `plugin_id`")
    _REGISTRY[cls.plugin_id] = cls
    return cls


def discover_plugins() -> None:
    """Import all modules inside `src.plugins.builtin` so they self-register."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    import src.plugins.builtin as builtin_pkg

    for module_info in pkgutil.iter_modules(builtin_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        mod_name = f"src.plugins.builtin.{module_info.name}"
        try:
            module = importlib.import_module(mod_name)
            # Also auto-register any AnimationPlugin subclass that forgot the decorator
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, AnimationPlugin) and obj is not AnimationPlugin and obj.plugin_id:
                    _REGISTRY.setdefault(obj.plugin_id, obj)
        except Exception:
            logger.exception("Failed to load plugin module %s", mod_name)
    _DISCOVERED = True


def get_plugin_class(plugin_id: str) -> Type[AnimationPlugin] | None:
    discover_plugins()
    return _REGISTRY.get(plugin_id)


def list_plugins_metadata() -> list[dict]:
    discover_plugins()
    return [cls.get_metadata() for _, cls in sorted(_REGISTRY.items())]
