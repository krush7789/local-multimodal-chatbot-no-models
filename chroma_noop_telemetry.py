"""No-op Chroma telemetry implementation to disable product telemetry calls."""

from __future__ import annotations

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetryClient(ProductTelemetryClient):
    """Drop all telemetry events without sending them externally."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
