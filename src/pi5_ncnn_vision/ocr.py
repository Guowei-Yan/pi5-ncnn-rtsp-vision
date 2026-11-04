from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pi5_ncnn_vision.types import DEFAULT_ALLOWED_SPEEDS, Detection

OCRPredictor = Callable[[list[np.ndarray]], Sequence[Any]]


def validate_speed_limit_text(
    text: str,
    confidence: float,
    min_confidence: float = 0.7,
    allowed_speeds: frozenset[int] = DEFAULT_ALLOWED_SPEEDS,
) -> bool:
    """Validate whether an OCR string represents a plausible numeric speed limit."""
    cleaned = text.strip()
    if confidence < min_confidence:
        return False
    if not cleaned.isdigit() or not (1 <= len(cleaned) <= 3):
        return False
    return int(cleaned) in allowed_speeds


def extract_rgb_crop(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Crop `[x1, y1, x2, y2]` from a BGR frame and convert to contiguous RGB."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 BGR frame, got shape {frame_bgr.shape}.")
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    x1, y1, x2, y2 = bbox
    x1i = max(round(x1), 0)
    y1i = max(round(y1), 0)
    x2i = min(round(x2), w)
    y2i = min(round(y2), h)

    if x2i <= x1i or y2i <= y1i:
        return None

    crop_bgr = frame_bgr[y1i:y2i, x1i:x2i, :3]
    if crop_bgr.size == 0:
        return None

    return np.ascontiguousarray(crop_bgr[..., ::-1])


class CropOCRProcessor:
    """
    Two-stage detection-to-OCR processor.

    Extracts bounding boxes matching `target_class_id` (e.g., `speed_limit_sign`),
    runs cropped RGB patches through a recognition predictor (such as `docTR`
    `crnn_mobilenet_v3_small`), and attaches validated OCR metadata to each detection.
    """

    def __init__(
        self,
        predictor: OCRPredictor | None = None,
        *,
        reco_arch: str = "crnn_mobilenet_v3_small",
        target_class_id: int = 1,
        min_det_score: float = 0.3,
        min_ocr_confidence: float = 0.7,
        max_crops_per_frame: int = 4,
        allowed_speeds: frozenset[int] = DEFAULT_ALLOWED_SPEEDS,
    ) -> None:
        if predictor is None:
            try:
                from doctr.models import recognition_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "python-doctr is required when no custom OCR predictor is supplied. "
                    "Install with: pip install 'pi5-ncnn-rtsp-vision[ocr]'"
                ) from exc
            predictor = recognition_predictor(reco_arch)

        self._predictor = predictor
        self.target_class_id = target_class_id
        self.min_det_score = min_det_score
        self.min_ocr_confidence = min_ocr_confidence
        self.max_crops_per_frame = max_crops_per_frame
        self.allowed_speeds = allowed_speeds

    @staticmethod
    def _parse_prediction(raw_item: Any) -> tuple[str, float]:
        if isinstance(raw_item, dict):
            return (
                str(raw_item.get("value", "")),
                float(raw_item.get("confidence", 0.0)),
            )
        if isinstance(raw_item, tuple) and len(raw_item) >= 2:
            return str(raw_item[0]), float(raw_item[1])
        return str(raw_item), 0.0

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        """Run OCR on up to `max_crops_per_frame` target-class detections."""
        if not detections or self.max_crops_per_frame <= 0:
            return list(detections)

        annotated = list(detections)
        candidate_indices: list[int] = []
        crops_rgb: list[np.ndarray] = []

        for idx, det in enumerate(annotated):
            if det.label != self.target_class_id or det.score < self.min_det_score:
                continue
            crop = extract_rgb_crop(frame_bgr, det.bbox)
            if crop is None:
                continue
            candidate_indices.append(idx)
            crops_rgb.append(crop)
            if len(crops_rgb) >= self.max_crops_per_frame:
                break

        if not crops_rgb:
            return annotated

        predictions = self._predictor(crops_rgb)
        for idx, raw_pred in zip(candidate_indices, predictions):
            text, conf = self._parse_prediction(raw_pred)
            accepted = validate_speed_limit_text(
                text=text,
                confidence=conf,
                min_confidence=self.min_ocr_confidence,
                allowed_speeds=self.allowed_speeds,
            )
            annotated[idx] = annotated[idx].with_ocr(
                text=text.strip(),
                confidence=conf,
                accepted=accepted,
            )

        return annotated
