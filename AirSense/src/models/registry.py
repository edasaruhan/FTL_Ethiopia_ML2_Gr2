"""Model Registry for AirSense ML lifecycle management.

Provides versioned storage, retrieval, and management of trained ML models
along with their metadata, parameters, and performance metrics.
"""

import json
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from src.core.exceptions import ModelError
from src.core.logging import get_logger


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class RegistryError(ModelError):
    """Raised for model registry-specific failures."""
    pass


# ---------------------------------------------------------------------------
# ModelMetadata dataclass  (Task 5.1)
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """Metadata for a registered model.

    Attributes:
        model_id: Unique identifier assigned at registration time.
        model_name: Human-readable name (e.g. "arima_PM2.5").
        model_type: Algorithm family (e.g. "arima", "lstm", "xgboost").
        version: Semantic version string (e.g. "1.0.0").
        file_path: Absolute path to the serialised model artifact.
        training_date: When the model was trained.
        parameters: Hyper-parameters used during training.
        metrics: Performance metrics produced during training/evaluation.
        tags: Arbitrary labels such as "production" or "experimental".
        target_pollutant: Pollutant the model was trained to forecast.
        training_data_source: Path or description of the training dataset.
        feature_set: Ordered list of feature column names used.
        created_at: When this registry entry was first created.
        updated_at: When this registry entry was last modified.
    """

    model_id: str
    model_name: str
    model_type: str
    version: str
    file_path: str
    training_date: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    target_pollutant: Optional[str]
    training_data_source: Optional[str]
    feature_set: Optional[List[str]]
    created_at: datetime
    updated_at: datetime

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Datetime fields are converted to ISO-8601 strings so the result
        can be written directly to a JSON file.

        Returns:
            Dictionary representation of this metadata object.
        """
        data = asdict(self)
        data["training_date"] = self.training_date.isoformat()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Reconstruct a ModelMetadata instance from a dictionary.

        ISO-8601 datetime strings are parsed back to :class:`datetime`
        objects automatically.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.

        Returns:
            Populated :class:`ModelMetadata` instance.
        """
        # Work on a copy so we don't mutate the caller's dict.
        d = dict(data)
        d["training_date"] = datetime.fromisoformat(d["training_date"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return cls(**d)


# ---------------------------------------------------------------------------
# ModelRegistry class  (Tasks 5.2 – 5.6)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """File-based model registry for managing ML model lifecycle.

    Directory layout managed by this class::

        <registry_dir>/
            registry.json          # Master index of all registered models
            <model_id>/
                model.joblib       # Serialised model artifact
                metadata.json      # Per-model metadata

    Args:
        registry_dir: Root directory for the registry.  Defaults to
            ``"models"``.  Created automatically if it does not exist.

    Raises:
        RegistryError: If the registry index cannot be loaded or saved.
    """

    def __init__(self, registry_dir: str = "models") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / "registry.json"
        self.logger = get_logger(__name__)

        # Ensure the root directory exists.
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index: model_id -> ModelMetadata
        self._registry: Dict[str, ModelMetadata] = {}
        self._load_registry()

    # ------------------------------------------------------------------
    # Task 5.3 – model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[List[str]] = None,
        target_pollutant: Optional[str] = None,
        training_data_source: Optional[str] = None,
        feature_set: Optional[List[str]] = None,
    ) -> str:
        """Register a trained model with its metadata.

        Creates a dedicated sub-directory under ``registry_dir``, saves the
        model artifact with :mod:`joblib`, writes a ``metadata.json`` file,
        and updates the master ``registry.json`` index.

        Args:
            model: The trained model object (must be serialisable by joblib).
            model_name: Human-readable name for the model.
            model_type: Algorithm family identifier (e.g. ``"arima"``).
            parameters: Hyper-parameters used during training.
            metrics: Performance metrics from training/evaluation.
            tags: Optional labels for categorisation (e.g. ``["production"]``).
            target_pollutant: Pollutant this model forecasts.
            training_data_source: Description or path of the training dataset.
            feature_set: Ordered list of feature column names.

        Returns:
            The unique ``model_id`` assigned to the registered model.

        Raises:
            RegistryError: If the model cannot be saved to disk.
        """
        model_id = self._generate_model_id(model_name, model_type)
        version = self._generate_version(model_name)

        # Create per-model directory.
        model_dir = self.registry_dir / model_id
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RegistryError(
                f"Cannot create model directory '{model_dir}': {exc}"
            ) from exc

        # Persist model artifact.
        model_path = model_dir / "model.joblib"
        try:
            joblib.dump(model, model_path)
        except Exception as exc:
            raise RegistryError(
                f"Failed to serialise model '{model_name}' to '{model_path}': {exc}"
            ) from exc

        now = datetime.now()
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            file_path=str(model_path),
            training_date=now,
            parameters=parameters,
            metrics=metrics,
            tags=tags or [],
            target_pollutant=target_pollutant,
            training_data_source=training_data_source,
            feature_set=feature_set,
            created_at=now,
            updated_at=now,
        )

        # Persist per-model metadata.
        metadata_path = model_dir / "metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(metadata.to_dict(), fh, indent=2)
        except OSError as exc:
            raise RegistryError(
                f"Failed to write metadata for model '{model_id}': {exc}"
            ) from exc

        # Update in-memory index and persist master registry.
        self._registry[model_id] = metadata
        self._save_registry()

        self.logger.info(
            "Model registered",
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            target_pollutant=target_pollutant,
        )

        return model_id

    # ------------------------------------------------------------------
    # Task 5.4 – model retrieval
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Retrieve a model artifact and its metadata.

        Args:
            model_id: Unique identifier returned by :meth:`register_model`.

        Returns:
            A ``(model_object, metadata)`` tuple.

        Raises:
            RegistryError: If ``model_id`` is not in the registry or the
                model file is missing from disk.
        """
        if model_id not in self._registry:
            raise RegistryError(
                f"Model '{model_id}' not found in registry. "
                "Use list_models() to see available models."
            )

        metadata = self._registry[model_id]
        model_path = Path(metadata.file_path)

        if not model_path.exists():
            raise RegistryError(
                f"Model file not found on disk: '{model_path}'. "
                "The registry entry exists but the artifact is missing."
            )

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise RegistryError(
                f"Failed to load model '{model_id}' from '{model_path}': {exc}"
            ) from exc

        self.logger.info("Model loaded from registry", model_id=model_id)
        return model, metadata

    # ------------------------------------------------------------------
    # Task 5.5 – listing and filtering
    # ------------------------------------------------------------------

    def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        target_pollutant: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """List registered models with optional filtering.

        All filters are applied with AND semantics; the ``tags`` filter uses
        OR semantics (any matching tag is sufficient).

        Args:
            model_type: If provided, only return models of this type.
            tags: If provided, only return models that have at least one of
                these tags.
            target_pollutant: If provided, only return models targeting this
                pollutant.

        Returns:
            List of :class:`ModelMetadata` objects sorted by creation date,
            newest first.
        """
        models: List[ModelMetadata] = list(self._registry.values())

        if model_type is not None:
            models = [m for m in models if m.model_type == model_type]

        if tags is not None:
            tag_set = set(tags)
            models = [m for m in models if tag_set.intersection(m.tags)]

        if target_pollutant is not None:
            models = [
                m for m in models if m.target_pollutant == target_pollutant
            ]

        # Newest first.
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    # ------------------------------------------------------------------
    # Task 5.6 – management operations
    # ------------------------------------------------------------------

    def delete_model(self, model_id: str) -> None:
        """Delete a model and all its associated files.

        Removes the model's sub-directory (artifact + metadata) and updates
        the master registry index.

        Args:
            model_id: Unique identifier of the model to delete.

        Raises:
            RegistryError: If ``model_id`` is not in the registry or the
                directory cannot be removed.
        """
        if model_id not in self._registry:
            raise RegistryError(
                f"Cannot delete: model '{model_id}' not found in registry."
            )

        metadata = self._registry[model_id]
        model_dir = Path(metadata.file_path).parent

        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
            except OSError as exc:
                raise RegistryError(
                    f"Failed to remove model directory '{model_dir}': {exc}"
                ) from exc

        del self._registry[model_id]
        self._save_registry()

        self.logger.info("Model deleted", model_id=model_id)

    def update_tags(self, model_id: str, tags: List[str]) -> None:
        """Replace the tags for a registered model.

        Updates both the in-memory index and the on-disk ``metadata.json``
        file, then persists the master registry.

        Args:
            model_id: Unique identifier of the model to update.
            tags: New list of tags (replaces existing tags entirely).

        Raises:
            RegistryError: If ``model_id`` is not in the registry or the
                metadata file cannot be written.
        """
        if model_id not in self._registry:
            raise RegistryError(
                f"Cannot update tags: model '{model_id}' not found in registry."
            )

        self._registry[model_id].tags = tags
        self._registry[model_id].updated_at = datetime.now()

        metadata_path = (
            Path(self._registry[model_id].file_path).parent / "metadata.json"
        )
        try:
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(self._registry[model_id].to_dict(), fh, indent=2)
        except OSError as exc:
            raise RegistryError(
                f"Failed to update metadata for model '{model_id}': {exc}"
            ) from exc

        self._save_registry()
        self.logger.info("Tags updated", model_id=model_id, tags=tags)

    def get_latest_version(self, model_name: str) -> Optional[ModelMetadata]:
        """Return the most recently registered version of a named model.

        Version comparison is lexicographic on the semantic version string,
        which works correctly for versions up to ``9.x.x``.  For larger
        version numbers callers should use :meth:`list_models` and sort
        manually.

        Args:
            model_name: The ``model_name`` to search for.

        Returns:
            The :class:`ModelMetadata` with the highest version, or ``None``
            if no model with that name is registered.
        """
        candidates = [
            m for m in self._registry.values() if m.model_name == model_name
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda m: m.version, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Task 5.2 – internal helpers
    # ------------------------------------------------------------------

    def _generate_model_id(self, model_name: str, model_type: str) -> str:
        """Generate a unique model identifier.

        The ID is composed of the model type, a sanitised model name, and a
        timestamp so that it is both human-readable and collision-resistant.

        Args:
            model_name: Human-readable model name.
            model_type: Algorithm family identifier.

        Returns:
            A string of the form ``"<type>_<name>_<YYYYMMDD_HHMMSS>"``.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitise name: keep only alphanumerics and underscores.
        safe_name = "".join(
            c if c.isalnum() or c == "_" else "_" for c in model_name
        )
        return f"{model_type}_{safe_name}_{timestamp}"

    def _generate_version(self, model_name: str) -> str:
        """Generate the next semantic version for a model name.

        Starts at ``"1.0.0"`` and increments the patch component for each
        subsequent registration of the same ``model_name``.

        Args:
            model_name: The model name whose version history to inspect.

        Returns:
            A semantic version string such as ``"1.0.3"``.
        """
        existing = [
            m for m in self._registry.values() if m.model_name == model_name
        ]

        if not existing:
            return "1.0.0"

        # Sort descending and increment the patch component of the latest.
        existing.sort(key=lambda m: m.version, reverse=True)
        latest = existing[0].version
        parts = latest.split(".")
        try:
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except (IndexError, ValueError):
            # Fallback: append a new minor version.
            return f"{latest}.1"

    def _load_registry(self) -> None:
        """Load the master registry index from ``registry.json``.

        If the file does not exist the in-memory registry is initialised as
        empty.  Corrupt or unreadable files are logged and treated as empty
        rather than raising, to avoid blocking startup.
        """
        if not self.registry_file.exists():
            self._registry = {}
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = json.load(fh)

            self._registry = {
                model_id: ModelMetadata.from_dict(metadata_dict)
                for model_id, metadata_dict in raw.items()
            }

            self.logger.info(
                "Registry loaded", model_count=len(self._registry)
            )

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failed to load registry; starting with empty registry",
                error=str(exc),
            )
            self._registry = {}

    def _save_registry(self) -> None:
        """Persist the master registry index to ``registry.json``.

        Raises:
            RegistryError: If the file cannot be written.
        """
        try:
            data = {
                model_id: metadata.to_dict()
                for model_id, metadata in self._registry.items()
            }
            with open(self.registry_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

            self.logger.debug("Registry saved", model_count=len(self._registry))

        except OSError as exc:
            raise RegistryError(
                f"Failed to save registry to '{self.registry_file}': {exc}"
            ) from exc
