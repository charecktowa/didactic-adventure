"""Checks that a model really works before the framework trains, registers or serves it.

Protocols and ABCs only check that methods exist. `validate_model` also checks their
signatures and runs the model on a small sample, so a broken model fails fast with a clear
message instead of halfway through a training run.
"""

import copy
import inspect
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_protocol_members

import numpy as np
import pandas as pd

from framework.contracts.model import Model, Trainable

# Arguments each contract method must accept, by name, besides `self` or `cls`.
_MODEL_PARAMETERS: Mapping[str, tuple[str, ...]] = {
    "predict": ("features",),
    "save": ("directory",),
    "load": ("directory",),
}
_TRAINABLE_PARAMETERS: Mapping[str, tuple[str, ...]] = {"fit": ("features", "target")}


class InvalidModelError(Exception):
    """Raised when a model does not fulfil the contracts; lists every problem found."""

    def __init__(self, model: object, problems: list[str]) -> None:
        self.problems = problems
        details = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"{type(model).__name__} is not a valid model:\n{details}")


def validate_model(
    model: object,
    features: pd.DataFrame,
    target: np.ndarray,
    *,
    trainable: bool = False,
) -> Model:
    """Check that ``model`` fulfils the contracts and works on a small sample.

    Set ``trainable`` when the caller is going to train the model: `fit` becomes required
    and is run on the sample. Otherwise the model is used as it is, like a pretrained one.
    The checks run on a copy, so ``model`` itself is never fitted or changed: the model must
    support `copy.deepcopy`.

    Raises:
        InvalidModelError: with every problem found.
    """
    problems = _check_structure(model, trainable=trainable)
    if not problems and isinstance(model, Model):
        problems = _check_behaviour(model, features, target, trainable=trainable)
        if not problems:
            return model
    raise InvalidModelError(model, problems)


def _check_structure(model: object, *, trainable: bool) -> list[str]:
    problems = _missing_methods(model, Model)
    if trainable:
        problems += _missing_methods(model, Trainable)
    if problems:
        return problems

    expected = dict(_MODEL_PARAMETERS)
    if isinstance(model, Trainable):
        expected |= _TRAINABLE_PARAMETERS
    return [
        problem
        for name, parameters in expected.items()
        if (problem := _check_signature(model, name, parameters))
    ]


def _missing_methods(model: object, contract: type) -> list[str]:
    return [
        f"missing method {name}() required by {contract.__name__}"
        for name in sorted(get_protocol_members(contract))
        if not callable(getattr(model, name, None))
    ]


def _check_signature(model: object, name: str, parameters: tuple[str, ...]) -> str | None:
    # Bound positionally: the framework calls contract methods positionally, so parameter
    # names are free (`fit(X, y)` is fine), as they are for mypy.
    try:
        signature = inspect.signature(getattr(model, name))
    except (TypeError, ValueError) as error:
        return f"{name}() has no inspectable signature: {error}"
    try:
        signature.bind(*parameters)
    except TypeError:
        return f"{name}{signature} must accept ({', '.join(parameters)})"
    return None


def _check_behaviour(
    model: Model, features: pd.DataFrame, target: np.ndarray, *, trainable: bool
) -> list[str]:
    try:
        model = copy.deepcopy(model)
    except Exception as error:
        return [
            f"cannot be copied for validation ({type(error).__name__}: {error}); "
            "define __deepcopy__ if it holds state that cannot be copied"
        ]
    try:
        if trainable and isinstance(model, Trainable):
            model.fit(features, target)
        predictions = model.predict(features)
        if np.shape(predictions) != (len(features),):
            return [
                f"predict() must return one value per row: expected shape ({len(features)},), "
                f"got {np.shape(predictions)}"
            ]
        with TemporaryDirectory() as directory:
            model.save(Path(directory))
            restored = type(model).load(Path(directory))
            if not np.array_equal(restored.predict(features), predictions):
                return ["predictions change after save() and load()"]
    except Exception as error:  # The model's own code failed: report it, don't propagate it.
        return [f"raised {type(error).__name__} while running: {error}"]
    return []
