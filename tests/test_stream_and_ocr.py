from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pi5_ncnn_vision.engine import HybridModelRouter
from pi5_ncnn_vision.ocr import (
    CropOCRProcessor,
    extract_rgb_crop,
    validate_speed_limit_text,
)
from pi5_ncnn_vision.stream import (
    FrameSlot,
    MultiStreamPipeline,
    RotatingNDJSONSink,
)
from pi5_ncnn_vision.types import Detection, PipelineConfig


def test_frame_slot_drop_oldest_semantics() -> None:
    slot = FrameSlot(stream_id=2)
    assert slot.take_if_new() is None

    frame1 = np.zeros((16, 16, 3), dtype=np.uint8)
    frame2 = np.ones((16, 16, 3), dtype=np.uint8) * 127
    frame3 = np.ones((16, 16, 3), dtype=np.uint8) * 255

    slot.put(frame1, capture_timestamp=1.0, frame_count=1)
    slot.put(frame2, capture_timestamp=2.0, frame_count=2)
    slot.put(frame3, capture_timestamp=3.0, frame_count=3)

    assert slot.overwrite_drops == 2
    packet = slot.take_if_new()
    assert packet is not None
    assert packet.frame_count == 3
    assert packet.stream_id == 2
    assert int(packet.frame_bgr[0, 0, 0]) == 255
    assert slot.take_if_new() is None


def test_rotating_ndjson_sink_rotation(tmp_path: Path) -> None:
    with RotatingNDJSONSink(
        tmp_path,
        enable_rotation=True,
        max_file_size_bytes=120,
        batch_size=2,
        prefix="test_events",
    ) as sink:
        for i in range(6):
            sink.write({"seq": i, "stream_id": i % 2, "msg": "edge_detection_event"})

    files = sorted(tmp_path.glob("test_events_*.ndjson"))
    assert len(files) >= 2
    records = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    assert [r["seq"] for r in records] == [0, 1, 2, 3, 4, 5]


def test_crop_ocr_processor_and_speed_validation() -> None:
    assert validate_speed_limit_text("60", 0.92) is True
    assert validate_speed_limit_text("110", 0.85) is True
    assert validate_speed_limit_text("65", 0.99) is False  # Not in allowed speed set
    assert validate_speed_limit_text("60", 0.50) is False  # Below min confidence
    assert validate_speed_limit_text("60km", 0.95) is False

    # Construct a BGR frame where blue=10, green=20, red=200 inside [10:30, 20:50]
    frame_bgr = np.zeros((60, 80, 3), dtype=np.uint8)
    frame_bgr[10:30, 20:50] = [10, 20, 200]

    crop_rgb = extract_rgb_crop(frame_bgr, (20.0, 10.0, 50.0, 30.0))
    assert crop_rgb is not None
    assert crop_rgb.shape == (20, 30, 3)
    # Verify BGR -> RGB channel conversion
    assert crop_rgb[0, 0].tolist() == [200, 20, 10]

    dets = [
        Detection(label=0, class_name="person", score=0.91, bbox=(0.0, 0.0, 15.0, 15.0)),
        Detection(
            label=1,
            class_name="speed_limit_sign",
            score=0.88,
            bbox=(20.0, 10.0, 50.0, 30.0),
        ),
    ]

    def fake_predictor(crops: list[np.ndarray]) -> list[dict[str, object]]:
        assert len(crops) == 1
        assert crops[0].shape == (20, 30, 3)
        return [{"value": "80", "confidence": 0.94}]

    processor = CropOCRProcessor(predictor=fake_predictor, target_class_id=1)
    enriched = processor.annotate(frame_bgr, dets)
    assert enriched[0].ocr_text is None
    assert enriched[1].ocr_text == "80"
    assert enriched[1].ocr_confidence == 0.94
    assert enriched[1].ocr_accepted is True


def test_multi_stream_pipeline_slot_processing(tmp_path: Path) -> None:
    class StubDetector:
        def __init__(self, label: int, name: str) -> None:
            self._label = label
            self._name = name

        def detect_timed(
            self, frame_bgr: np.ndarray
        ) -> tuple[list[Detection], float, float, float]:
            return (
                [
                    Detection(
                        label=self._label,
                        class_name=self._name,
                        score=0.9,
                        bbox=(5.0, 5.0, 25.0, 25.0),
                    )
                ],
                1.0,
                12.0,
                0.5,
            )

        def close(self) -> None:
            pass

    cfg = PipelineConfig.from_mapping(
        {
            "rtsp_urls": [
                "rtsp://192.0.2.1/s0",
                "rtsp://192.0.2.2/s1",
                "rtsp://192.0.2.3/s2",
            ],
            "model_param_cpu": "cpu.param",
            "model_bin_cpu": "cpu.bin",
            "gpu_stream_id": 1,
            "output_dir": str(tmp_path),
        }
    )

    router = HybridModelRouter(
        cpu_detector=StubDetector(1, "speed_limit_sign"),  # type: ignore[arg-type]
        gpu_detector=StubDetector(1, "speed_limit_sign"),  # type: ignore[arg-type]
        gpu_stream_id=1,
    )
    ocr = CropOCRProcessor(
        predictor=lambda crops: [{"value": "60", "confidence": 0.96} for _ in crops],
        target_class_id=1,
    )
    pipeline = MultiStreamPipeline(cfg, router, ocr_processor=ocr)

    dummy_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    pipeline.slots[0].put(dummy_frame, capture_timestamp=10.0, frame_count=1)
    pipeline.slots[1].put(dummy_frame, capture_timestamp=10.0, frame_count=1)

    records = pipeline.process_available_slots_once()
    assert len(records) == 2
    assert records[0].backend == "CPU"
    assert records[1].backend == "GPU"
    assert records[0].detections[0].ocr_text == "60"
    assert records[0].detections[0].ocr_accepted is True

    stats = pipeline.snapshot_stats()
    assert stats["num_streams"] == 3
    assert stats["total_inferences"] == 2
    assert stats["total_detections"] == 2
