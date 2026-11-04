from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pi5_ncnn_vision.types import DecoderSpec, Detection


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the given axis."""
    x_f32 = x.astype(np.float32, copy=False)
    shifted = x_f32 - np.max(x_f32, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def compute_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Vectorized intersection-over-union between two sets of [x1, y1, x2, y2] boxes."""
    lt = np.maximum(boxes_a[..., :2], boxes_b[..., :2])
    rb = np.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[..., 2] - boxes_a[..., 0], 0.0, None) * np.clip(
        boxes_a[..., 3] - boxes_a[..., 1], 0.0, None
    )
    area_b = np.clip(boxes_b[..., 2] - boxes_b[..., 0], 0.0, None) * np.clip(
        boxes_b[..., 3] - boxes_b[..., 1], 0.0, None
    )
    return inter / (area_a + area_b - inter + eps)


def hard_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    top_k: int = -1,
) -> np.ndarray:
    """Greedy hard Non-Maximum Suppression returning kept row indices."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(scores, kind="mergesort")
    picked: list[int] = []

    while order.size > 0:
        current = int(order[-1])
        picked.append(current)
        if (0 < top_k == len(picked)) or order.size == 1:
            break
        rest = order[:-1]
        ious = compute_iou(boxes[rest], boxes[current][None, :]).reshape(-1)
        order = rest[ious <= iou_threshold]

    return np.asarray(picked, dtype=np.int64)


class PicoDetDecoder:
    """
    Host-side multi-scale PicoHeadV2 decoder for NCNN models exported with
    `export.post_process=False` (stripped `MultiClassNMS` / `MatrixNMS`).

    Converts 4-scale classification scores and Distribution Focal Loss (DFL)
    regression logits into rescaled, NMS-filtered bounding boxes.
    """

    def __init__(self, spec: DecoderSpec | None = None) -> None:
        self.spec = spec or DecoderSpec()
        self._reg_bins = np.arange(self.spec.reg_max + 1, dtype=np.float32)
        self._reg_dim = 4 * (self.spec.reg_max + 1)
        self._direction_signs = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)

        self._centers_by_length: dict[int, np.ndarray] = {}
        self._stride_by_length: dict[int, float] = {}
        for stride in self.spec.fpn_strides:
            feat_h = self.spec.input_height // stride
            feat_w = self.spec.input_width // stride
            yy, xx = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing="ij",
            )
            cx = (xx.reshape(-1) + 0.5) * float(stride)
            cy = (yy.reshape(-1) + 0.5) * float(stride)
            centers = np.stack([cx, cy, cx, cy], axis=1).astype(np.float32)
            num_anchors = feat_h * feat_w
            self._centers_by_length[num_anchors] = centers
            self._stride_by_length[num_anchors] = float(stride)

    @property
    def reg_dim(self) -> int:
        return self._reg_dim

    @property
    def expected_anchor_counts(self) -> tuple[int, ...]:
        return tuple(sorted(self._centers_by_length.keys(), reverse=True))

    @staticmethod
    def reshape_head_blob(blob: np.ndarray, feature_dim: int) -> np.ndarray:
        """Normalize an NCNN Mat array of shape (L, F), (F, L), or 1D into (L, F)."""
        arr = np.asarray(blob, dtype=np.float32)
        if arr.ndim == 2:
            rows, cols = arr.shape
            if cols == feature_dim:
                return arr
            if rows == feature_dim:
                return np.ascontiguousarray(arr.T)
            if arr.size % feature_dim == 0:
                return arr.reshape(-1, feature_dim)
            raise ValueError(
                f"Cannot reshape 2D blob of shape {arr.shape} into (*, {feature_dim})."
            )
        if arr.size > 0 and arr.size % feature_dim == 0:
            return arr.reshape(-1, feature_dim)
        raise ValueError(
            f"Cannot reshape blob of shape {arr.shape} into (*, {feature_dim})."
        )

    def pair_multiscale_blobs(
        self,
        raw_blobs: Mapping[str, np.ndarray],
        cls_names: Sequence[str] = ("out0", "out1", "out2", "out3"),
        reg_names: Sequence[str] = ("out4", "out5", "out6", "out7"),
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Pair classification and DFL regression blobs by matching spatial anchor count L."""
        cls_by_length: dict[int, np.ndarray] = {}
        reg_by_length: dict[int, np.ndarray] = {}

        for name in cls_names:
            if name not in raw_blobs:
                raise KeyError(f"Missing classification output blob '{name}'.")
            cls_mat = self.reshape_head_blob(raw_blobs[name], self.spec.num_classes)
            cls_by_length[cls_mat.shape[0]] = cls_mat

        for name in reg_names:
            if name not in raw_blobs:
                raise KeyError(f"Missing regression output blob '{name}'.")
            reg_mat = self.reshape_head_blob(raw_blobs[name], self._reg_dim)
            reg_by_length[reg_mat.shape[0]] = reg_mat

        paired: list[tuple[int, np.ndarray, np.ndarray]] = []
        for num_anchors, cls_mat in cls_by_length.items():
            if num_anchors not in reg_by_length:
                raise ValueError(
                    f"No matching regression head found for anchor count L={num_anchors}."
                )
            if num_anchors not in self._centers_by_length:
                raise ValueError(
                    f"Anchor count L={num_anchors} does not match configured FPN strides "
                    f"{self.spec.fpn_strides} for input {self.spec.input_width}x{self.spec.input_height}."
                )
            paired.append((num_anchors, cls_mat, reg_by_length[num_anchors]))

        paired.sort(key=lambda item: item[0], reverse=True)
        return paired

    def decode_dfl_boxes(
        self,
        reg_logits: np.ndarray,
        anchor_indices: np.ndarray,
        num_anchors: int,
    ) -> np.ndarray:
        """
        Project DFL distribution logits [K, 4 * (reg_max + 1)] into [K, 4] bounding
        boxes in model input pixel coordinates via softmax integral projection.
        """
        selected = reg_logits[anchor_indices, :].reshape(-1, self.spec.reg_max + 1)
        probs = softmax(selected, axis=1)
        distances = (probs @ self._reg_bins).reshape(-1, 4)
        distances *= self._stride_by_length[num_anchors]
        centers = self._centers_by_length[num_anchors][anchor_indices, :]
        return centers + self._direction_signs * distances

    def decode(
        self,
        paired_levels: Sequence[tuple[int, np.ndarray, np.ndarray]],
        orig_size: tuple[int, int],
    ) -> list[Detection]:
        """
        Decode paired FPN levels into sorted `Detection` objects in original image
        coordinates `(orig_height, orig_width)`.
        """
        orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid original image dimensions: {orig_size}.")

        active_classes = (
            self.spec.target_class_ids
            if self.spec.target_class_ids is not None
            else tuple(range(self.spec.num_classes))
        )

        scale_x = float(orig_w) / float(self.spec.input_width)
        scale_y = float(orig_h) / float(self.spec.input_height)
        max_x = float(max(0, orig_w - 1))
        max_y = float(max(0, orig_h - 1))

        detections: list[Detection] = []

        for class_id in active_classes:
            class_boxes: list[np.ndarray] = []
            class_scores: list[np.ndarray] = []

            for num_anchors, cls_mat, reg_mat in paired_levels:
                scores = cls_mat[:, class_id].astype(np.float32, copy=False)
                idx = np.flatnonzero(scores >= self.spec.score_threshold)
                if idx.size == 0:
                    continue
                if 0 < self.spec.nms_top_k < idx.size:
                    level_scores = scores[idx]
                    part = np.argpartition(
                        level_scores, level_scores.size - self.spec.nms_top_k
                    )[-self.spec.nms_top_k :]
                    idx = idx[part]

                boxes = self.decode_dfl_boxes(reg_mat, idx, num_anchors)
                class_boxes.append(boxes)
                class_scores.append(scores[idx])

            if not class_boxes:
                continue

            boxes_all = np.concatenate(class_boxes, axis=0)
            scores_all = np.concatenate(class_scores, axis=0)

            if 0 < self.spec.nms_top_k < scores_all.size:
                top_idx = np.argpartition(
                    scores_all, scores_all.size - self.spec.nms_top_k
                )[-self.spec.nms_top_k :]
                boxes_all = boxes_all[top_idx]
                scores_all = scores_all[top_idx]

            boxes_all[:, 0] = np.clip(boxes_all[:, 0] * scale_x, 0.0, max_x)
            boxes_all[:, 2] = np.clip(boxes_all[:, 2] * scale_x, 0.0, max_x)
            boxes_all[:, 1] = np.clip(boxes_all[:, 1] * scale_y, 0.0, max_y)
            boxes_all[:, 3] = np.clip(boxes_all[:, 3] * scale_y, 0.0, max_y)

            keep_indices = hard_nms(
                boxes_all,
                scores_all,
                iou_threshold=self.spec.nms_iou_threshold,
                top_k=self.spec.keep_top_k,
            )

            class_name = self.spec.resolve_class_name(class_id)
            for idx_kept in keep_indices:
                b = boxes_all[int(idx_kept)]
                detections.append(
                    Detection(
                        label=int(class_id),
                        class_name=class_name,
                        score=float(scores_all[int(idx_kept)]),
                        bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    )
                )

        detections.sort(key=lambda d: d.score, reverse=True)
        if 0 < self.spec.keep_top_k < len(detections):
            detections = detections[: self.spec.keep_top_k]
        return detections
