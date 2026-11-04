from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "probesize;32|"
    "analyzeduration;0|"
    "stimeout;5000000"
)

DEFAULT_ALLOWED_SPEEDS: frozenset[int] = frozenset(
    {20, 30, 40, 50, 60, 70, 80, 90, 100, 110}
)


@dataclass(frozen=True, slots=True)
class Detection:
    """Single object detection in original image pixel coordinates."""

    label: int
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    ocr_accepted: bool | None = None

    def with_ocr(
        self,
        *,
        text: str,
        confidence: float,
        accepted: bool,
    ) -> Detection:
        return Detection(
            label=self.label,
            class_name=self.class_name,
            score=self.score,
            bbox=self.bbox,
            ocr_text=text,
            ocr_confidence=confidence,
            ocr_accepted=accepted,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "class_name": self.class_name,
            "score": round(float(self.score), 6),
            "bbox": [round(float(coord), 2) for coord in self.bbox],
        }
        if self.ocr_text is not None:
            payload["ocr"] = {
                "text": self.ocr_text,
                "confidence": round(float(self.ocr_confidence or 0.0), 6),
                "accepted": bool(self.ocr_accepted),
            }
        return payload


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Latest unconsumed frame sampled from an RTSP stream."""

    stream_id: int
    frame_count: int
    capture_timestamp: float
    frame_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    """Complete telemetry and detection record for one processed frame."""

    stream_id: int
    frame_count: int
    capture_timestamp: float
    inference_timestamp: float
    latency_ms: float
    preprocess_ms: float
    forward_ms: float
    postprocess_ms: float
    ocr_ms: float
    inference_time_ms: float
    backend: str
    detections: tuple[Detection, ...]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "frame_count": self.frame_count,
            "capture_timestamp": round(self.capture_timestamp, 6),
            "inference_timestamp": round(self.inference_timestamp, 6),
            "latency_ms": round(self.latency_ms, 3),
            "inference_time_ms": round(self.inference_time_ms, 3),
            "timing_breakdown_ms": {
                "preprocess": round(self.preprocess_ms, 3),
                "forward": round(self.forward_ms, 3),
                "postprocess": round(self.postprocess_ms, 3),
                "ocr": round(self.ocr_ms, 3),
            },
            "backend": self.backend,
            "detection_count": self.detection_count,
            "detections": [det.to_dict() for det in self.detections],
        }


@dataclass(frozen=True, slots=True)
class DecoderSpec:
    """Configuration for stripped-postprocess PicoHeadV2 / DFL multi-scale decoding."""

    input_width: int = 320
    input_height: int = 320
    fpn_strides: tuple[int, ...] = (8, 16, 32, 64)
    num_classes: int = 80
    reg_max: int = 7
    score_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    nms_top_k: int = 1000
    keep_top_k: int = 100
    target_class_ids: tuple[int, ...] | None = (0,)
    class_names: tuple[str, ...] = ("person",)

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("input_width and input_height must be positive integers.")
        if not self.fpn_strides or any(s <= 0 for s in self.fpn_strides):
            raise ValueError("fpn_strides must contain positive integers.")
        for s in self.fpn_strides:
            if self.input_width % s != 0 or self.input_height % s != 0:
                raise ValueError(
                    f"Input dimensions ({self.input_width}x{self.input_height}) "
                    f"must be divisible by FPN stride {s}."
                )
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.reg_max < 1:
            raise ValueError("reg_max must be >= 1.")
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError("score_threshold must be within [0.0, 1.0].")
        if not (0.0 < self.nms_iou_threshold <= 1.0):
            raise ValueError("nms_iou_threshold must be within (0.0, 1.0].")
        if self.target_class_ids is not None:
            for cid in self.target_class_ids:
                if cid < 0 or cid >= self.num_classes:
                    raise ValueError(
                        f"target_class_id {cid} out of range for num_classes={self.num_classes}."
                    )

    def resolve_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for the multi-stream RTSP NCNN inference pipeline."""

    rtsp_urls: tuple[str, ...]
    model_param_cpu: str
    model_bin_cpu: str
    model_param_gpu: str | None = None
    model_bin_gpu: str | None = None
    decoder: DecoderSpec = field(default_factory=DecoderSpec)
    target_fps_per_stream: float = 2.0
    cap_buffersize: int = 1
    result_queue_maxsize: int = 100
    ncnn_num_threads: int = 2
    use_hybrid_mode: bool = True
    gpu_device_id: int = 0
    gpu_stream_id: int | None = None
    gpu_validation_runs: int = 5
    opencv_ffmpeg_options: str = DEFAULT_FFMPEG_CAPTURE_OPTIONS
    reconnect_delay_sec: float = 5.0
    output_dir: str = "output"
    save_images: bool = False
    enable_log_rotation: bool = True
    max_json_file_size_mb: float = 100.0
    json_batch_size: int = 10
    json_flush_interval_sec: float = 1.0
    enable_sign_ocr: bool = False
    sign_class_id: int = 1
    sign_score_threshold: float = 0.3
    ocr_min_confidence: float = 0.7
    ocr_max_crops_per_frame: int = 4
    ocr_reco_arch: str = "crnn_mobilenet_v3_small"
    allowed_speeds: frozenset[int] = DEFAULT_ALLOWED_SPEEDS

    def __post_init__(self) -> None:
        if not self.rtsp_urls:
            raise ValueError("At least one RTSP URL must be configured in rtsp_urls.")
        if not self.model_param_cpu or not self.model_bin_cpu:
            raise ValueError("model_param_cpu and model_bin_cpu are required.")
        if self.target_fps_per_stream <= 0:
            raise ValueError("target_fps_per_stream must be > 0.")
        if self.ncnn_num_threads < 1:
            raise ValueError("ncnn_num_threads must be >= 1.")
        if self.gpu_stream_id is not None and not (0 <= self.gpu_stream_id < len(self.rtsp_urls)):
            raise ValueError(
                f"gpu_stream_id={self.gpu_stream_id} must be within [0, {len(self.rtsp_urls) - 1}]."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PipelineConfig:
        urls_raw = data.get("rtsp_urls", ())
        if isinstance(urls_raw, str):
            urls = (urls_raw,)
        else:
            urls = tuple(str(u) for u in urls_raw)

        target_ids_raw = data.get("target_class_ids", (0,))
        target_ids = (
            None
            if target_ids_raw is None
            else tuple(int(x) for x in target_ids_raw)
        )
        class_names_raw = data.get("class_names", ("person",))
        class_names = tuple(str(c) for c in class_names_raw)

        decoder = DecoderSpec(
            input_width=int(data.get("input_width", 320)),
            input_height=int(data.get("input_height", 320)),
            fpn_strides=tuple(int(s) for s in data.get("fpn_strides", (8, 16, 32, 64))),
            num_classes=int(data.get("num_classes", 80)),
            reg_max=int(data.get("reg_max", 7)),
            score_threshold=float(data.get("score_threshold", 0.5)),
            nms_iou_threshold=float(data.get("nms_iou_threshold", 0.5)),
            nms_top_k=int(data.get("nms_top_k", 1000)),
            keep_top_k=int(data.get("keep_top_k", 100)),
            target_class_ids=target_ids,
            class_names=class_names,
        )

        allowed_raw = data.get("allowed_speeds")
        if allowed_raw is None:
            allowed_speeds = DEFAULT_ALLOWED_SPEEDS
        elif isinstance(allowed_raw, str):
            allowed_speeds = frozenset(
                int(part.strip()) for part in allowed_raw.split(",") if part.strip()
            )
        else:
            allowed_speeds = frozenset(int(v) for v in allowed_raw)

        model_param_cpu = str(data.get("model_param_cpu", ""))
        model_bin_cpu = str(data.get("model_bin_cpu", ""))

        return cls(
            rtsp_urls=urls,
            model_param_cpu=model_param_cpu,
            model_bin_cpu=model_bin_cpu,
            model_param_gpu=str(data.get("model_param_gpu") or model_param_cpu),
            model_bin_gpu=str(data.get("model_bin_gpu") or model_bin_cpu),
            decoder=decoder,
            target_fps_per_stream=float(data.get("target_fps_per_stream", 2.0)),
            cap_buffersize=int(data.get("cap_buffersize", 1)),
            result_queue_maxsize=int(data.get("result_queue_maxsize", 100)),
            ncnn_num_threads=int(data.get("ncnn_num_threads", 2)),
            use_hybrid_mode=bool(data.get("use_hybrid_mode", True)),
            gpu_device_id=int(data.get("gpu_device_id", 0)),
            gpu_stream_id=(
                None
                if data.get("gpu_stream_id") is None
                else int(data["gpu_stream_id"])
            ),
            gpu_validation_runs=int(data.get("gpu_validation_runs", 5)),
            opencv_ffmpeg_options=str(
                data.get("opencv_ffmpeg_options", DEFAULT_FFMPEG_CAPTURE_OPTIONS)
            ),
            reconnect_delay_sec=float(data.get("reconnect_delay_sec", 5.0)),
            output_dir=str(data.get("output_dir", "output")),
            save_images=bool(data.get("save_images", False)),
            enable_log_rotation=bool(data.get("enable_log_rotation", True)),
            max_json_file_size_mb=float(data.get("max_json_file_size_mb", 100.0)),
            json_batch_size=int(data.get("json_batch_size", 10)),
            json_flush_interval_sec=float(data.get("json_flush_interval_sec", 1.0)),
            enable_sign_ocr=bool(data.get("enable_sign_ocr", False)),
            sign_class_id=int(data.get("sign_class_id", 1)),
            sign_score_threshold=float(data.get("sign_score_threshold", 0.3)),
            ocr_min_confidence=float(data.get("ocr_min_confidence", 0.7)),
            ocr_max_crops_per_frame=int(data.get("ocr_max_crops_per_frame", 4)),
            ocr_reco_arch=str(data.get("ocr_reco_arch", "crnn_mobilenet_v3_small")),
            allowed_speeds=allowed_speeds,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}.")
        return cls.from_mapping(raw)
