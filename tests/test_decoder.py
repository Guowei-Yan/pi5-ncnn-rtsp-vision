from __future__ import annotations

import numpy as np
import pytest

from pi5_ncnn_vision.decoder import PicoDetDecoder, compute_iou, hard_nms, softmax
from pi5_ncnn_vision.types import DecoderSpec


def test_softmax_numerical_stability() -> None:
    logits = np.array([[1000.0, 1000.0, 1000.0], [-1000.0, -999.0, -1000.0]], dtype=np.float32)
    probs = softmax(logits, axis=1)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(probs[0], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)
    assert probs[1, 1] > probs[1, 0]


def test_compute_iou_and_hard_nms() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 110.0, 110.0],
            [12.0, 12.0, 112.0, 112.0],  # High overlap with box 0
            [200.0, 200.0, 300.0, 300.0],  # Disjoint box
        ],
        dtype=np.float32,
    )
    scores = np.array([0.95, 0.80, 0.70], dtype=np.float32)

    iou_01 = compute_iou(boxes[0:1], boxes[1:2]).item()
    assert 0.85 < iou_01 < 1.0

    kept = hard_nms(boxes, scores, iou_threshold=0.5)
    assert kept.tolist() == [0, 2]

    kept_top1 = hard_nms(boxes, scores, iou_threshold=0.5, top_k=1)
    assert kept_top1.tolist() == [0]


def test_decoder_dfl_integral_projection_and_rescaling() -> None:
    spec = DecoderSpec(
        input_width=320,
        input_height=320,
        fpn_strides=(8, 16, 32, 64),
        num_classes=80,
        reg_max=7,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        target_class_ids=(0,),
        class_names=("person",),
    )
    decoder = PicoDetDecoder(spec)
    assert decoder.expected_anchor_counts == (1600, 400, 100, 25)

    raw_blobs: dict[str, np.ndarray] = {}
    anchor_counts = [1600, 400, 100, 25]
    for idx, count in enumerate(anchor_counts):
        raw_blobs[f"out{idx}"] = np.zeros((count, 80), dtype=np.float32)
        raw_blobs[f"out{idx + 4}"] = np.full((count, 32), -20.0, dtype=np.float32)

    # Activate a single anchor on stride=32 (10x10 grid -> L=100) at row=5, col=5 (index 55)
    # Grid center in 320x320 space: cx = (5 + 0.5) * 32 = 176.0, cy = (5 + 0.5) * 32 = 176.0
    anchor_idx = 5 * 10 + 5
    raw_blobs["out2"][anchor_idx, 0] = 0.92

    # Set DFL bins strongly to: left=2, top=3, right=2, bottom=4 (in stride units of 32)
    # Expected box in 320x320 space:
    # x1 = 176 - 2*32 = 112, y1 = 176 - 3*32 = 80, x2 = 176 + 2*32 = 240, y2 = 176 + 4*32 = 304
    reg_target = raw_blobs["out6"][anchor_idx].reshape(4, 8)
    reg_target[0, 2] = 20.0
    reg_target[1, 3] = 20.0
    reg_target[2, 2] = 20.0
    reg_target[3, 4] = 20.0

    paired = decoder.pair_multiscale_blobs(raw_blobs)
    # Original frame is 640x960 (height=640 -> scale_y=2.0, width=960 -> scale_x=3.0)
    detections = decoder.decode(paired, orig_size=(640, 960))
    assert len(detections) == 1
    det = detections[0]
    assert det.label == 0
    assert det.class_name == "person"
    assert pytest.approx(det.score, rel=1e-5) == 0.92
    np.testing.assert_allclose(
        det.bbox,
        (112.0 * 3.0, 80.0 * 2.0, 240.0 * 3.0, 304.0 * 2.0),
        atol=1e-2,
    )


def test_decoder_two_class_and_transposed_blobs() -> None:
    spec = DecoderSpec(
        input_width=640,
        input_height=640,
        fpn_strides=(8, 16, 32, 64),
        num_classes=2,
        reg_max=7,
        score_threshold=0.4,
        target_class_ids=(0, 1),
        class_names=("person", "speed_limit_sign"),
    )
    decoder = PicoDetDecoder(spec)
    assert decoder.expected_anchor_counts == (6400, 1600, 400, 100)

    raw_blobs: dict[str, np.ndarray] = {}
    for idx, count in enumerate(decoder.expected_anchor_counts):
        # Pass transposed (F, L) classification blobs to verify reshape_head_blob
        raw_blobs[f"out{idx}"] = np.zeros((2, count), dtype=np.float32)
        raw_blobs[f"out{idx + 4}"] = np.zeros((count, 32), dtype=np.float32)

    raw_blobs["out2"][0, 10] = 0.85  # person on L=400
    raw_blobs["out3"][1, 20] = 0.91  # speed_limit_sign on L=100

    paired = decoder.pair_multiscale_blobs(raw_blobs)
    dets = decoder.decode(paired, orig_size=(640, 640))
    assert len(dets) == 2
    assert [d.class_name for d in dets] == ["speed_limit_sign", "person"]


def test_decoder_spec_validation() -> None:
    with pytest.raises(ValueError, match="divisible by FPN stride"):
        DecoderSpec(input_width=300, input_height=300, fpn_strides=(8, 16, 32, 64))
