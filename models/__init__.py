"""Model recipes: each package here has a `model.py` whose `build_model()` returns a model."""

import importlib
import pkgutil

from framework.contracts.model import Model


def available_models() -> list[str]:
    """Names of the model packages in this directory."""
    return sorted(module.name for module in pkgutil.iter_modules(__path__) if module.ispkg)


def build(name: str) -> Model:
    """Build the model of the package called ``name``, with its default settings."""
    if name not in available_models():
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(available_models())}")
    recipe = importlib.import_module(f"{__name__}.{name}.model")
    model: Model = recipe.build_model()
    return model
