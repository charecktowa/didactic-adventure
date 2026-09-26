"""ZenML materializer that stores models with their own `save()` and `load()`."""

import importlib
import json
import os
from pathlib import Path
from typing import Any, ClassVar

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import io_utils

from framework.connectors.base import ModelConnector
from framework.contracts.model import Model

MODEL_DIR = "model"
MODEL_CLASS_FILE = "model_class.json"


class ModelMaterializer(BaseMaterializer):
    """Stores any `Model` in the format it writes itself, instead of pickling the object.

    `model/` holds exactly what `save()` writes, so it also loads without ZenML.
    `model_class.json` names the class that wrote it: steps usually ask for the `Model`
    protocol, which cannot load anything by itself.
    """

    # Registers the materializer for every connector; other models get it per step.
    ASSOCIATED_TYPES: ClassVar[tuple[type[Any], ...]] = (ModelConnector,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.MODEL

    @classmethod
    def can_save_type(cls, data_type: type[Any]) -> bool:
        """Accept any class that fulfils the `Model` contract, connector or not."""
        return issubclass(data_type, Model)

    @classmethod
    def can_load_type(cls, data_type: type[Any]) -> bool:
        """Accept any class that fulfils the `Model` contract, connector or not."""
        return issubclass(data_type, Model)

    def save(self, data: Model) -> None:
        """Write the model with its own `save()`, then copy it to the artifact store."""
        model_class = type(data)
        io_utils.create_dir_recursive_if_not_exists(self.uri)
        with fileio.open(os.path.join(self.uri, MODEL_CLASS_FILE), "w") as file:
            json.dump(
                {"module": model_class.__module__, "qualname": model_class.__qualname__}, file
            )
        with self.get_temporary_directory(delete_at_exit=True) as directory:
            data.save(Path(directory))
            io_utils.copy_dir(directory, os.path.join(self.uri, MODEL_DIR))

    def load(self, data_type: type[Any]) -> Model:
        """Copy the artifact from the store, then read it with the saving class's `load()`."""
        model_class = self._saved_class()
        if not issubclass(model_class, data_type):
            raise TypeError(f"{self.uri} holds a {model_class.__qualname__}, not a {data_type}")
        with self.get_temporary_directory(delete_at_exit=True) as directory:
            io_utils.copy_dir(os.path.join(self.uri, MODEL_DIR), directory)
            model: Model = model_class.load(Path(directory))
            return model

    def _saved_class(self) -> type[Model]:
        # The artifact store is trusted, as it is for ZenML's own pickle materializer: the
        # module named here is imported. What it names must still be a `Model` class.
        path = os.path.join(self.uri, MODEL_CLASS_FILE)
        try:
            with fileio.open(path, "r") as file:
                saved = json.load(file)
            module, qualname = saved["module"], saved["qualname"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{path} does not name the class that saved the model") from error

        model_class: Any = importlib.import_module(module)
        for attribute in qualname.split("."):
            model_class = getattr(model_class, attribute)
        if not (isinstance(model_class, type) and issubclass(model_class, Model)):
            raise TypeError(f"{path} names {module}.{qualname}, which is not a Model class")
        return model_class
