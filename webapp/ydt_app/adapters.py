"""Adapters between learned observable models and downstream metabolism.

The generic wet-lab workflow deliberately has no observable-to-GSM calibration
yet. Keeping the boundary explicit prevents the web app from treating a direct
product predictor as an exact Yeast9 hybrid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetabolicInterfaceAdapter(ABC):
    name = "unconfigured"

    @abstractmethod
    def supports_schema(self, schema: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> dict[str, Any]:
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError("No generic observable-to-metabolic adapter is configured.")

    def predict_controls(self, *args, **kwargs):
        raise NotImplementedError("No generic observable-to-metabolic adapter is configured.")


class CanonicalSyntheticInterfaceAdapter(MetabolicInterfaceAdapter):
    """Placeholder boundary for the repository's canonical synthetic mode.

    It is intentionally not accepted by arbitrary uploaded wet-lab schemas.
    The Yeast9 validation implementation remains in the notebook support package,
    not a generic browser inference service.
    """

    name = "canonical synthetic interface"

    def supports_schema(self, schema: dict[str, Any]) -> bool:
        required = {"temperature", "pH", "DO", "product"}
        return required.issubset(set(schema.get("environment", [])) | {schema.get("product")})

    def validate(self) -> dict[str, Any]:
        return {"available": False, "status": "adapter boundary only", "reason": "Canonical synthetic Yeast9 execution is not exposed through the arbitrary-upload workflow."}
