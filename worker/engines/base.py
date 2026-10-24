"""Engine interface every image-to-3D adapter implements (blueprint P1-04)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image


class Cancelled(Exception):
    """The user cancelled the job; stop quietly at the next stage boundary."""


class EngineError(Exception):
    """A failure with a stable error code shown to the user (e.g. INFERENCE_OOM, SEGMENTATION_EMPTY)."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class Reporter(Protocol):
    def stage(self, status: str, stage: str, progress: float | None = None) -> None:
        """Report a job status (preparing/running/postprocessing) and a finer stage name. Raises Cancelled."""

    def check_cancel(self) -> None:
        """Raise Cancelled if the user cancelled. Call between expensive steps."""


@dataclass
class EngineResult:
    glb: bytes
    # Reproducibility record merged into manifest.json: engine commit, weights revision,
    # seed actually used, per-stage timings, peak VRAM, segmenter and its licence, ...
    metadata: dict = field(default_factory=dict)
    gpu_seconds: float | None = None


class Engine(ABC):
    id: str
    textured: bool = True  # output must carry base colour + metallic-roughness textures (gate V4)

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the device. Called once per worker boot."""

    @abstractmethod
    def run(self, image: Image.Image, params: dict, report: Reporter) -> EngineResult:
        """Turn one sanitised RGB/RGBA image into a GLB. Must honour params['seed']."""
