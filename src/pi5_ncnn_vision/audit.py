from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NCNN_RISKY_ONNX_OPS: frozenset[str] = frozenset(
    {
        "NonMaxSuppression",
        "NonMaxSuppressionV3",
        "NonMaxSuppressionV5",
        "NonMaxSuppressionV9",
        "MulticlassNMS",
        "MultiClassNMS",
        "MatrixNMS",
        "GatherND",
        "ScatterND",
        "Einsum",
        "Loop",
        "If",
        "Scan",
        "Where",
        "DepthToSpace",
        "SpaceToDepth",
        "GridSample",
        "GridSampleV2",
        "RoiAlign",
        "RoiPool",
        "NonZero",
        "Unique",
        "DynamicSlice",
        "DynamicGather",
        "TopK",
    }
)


@dataclass(frozen=True, slots=True)
class OnnxAuditReport:
    """Summary of ONNX graph compatibility with `onnx2ncnn` conversion."""

    total_nodes: int
    op_counts: dict[str, int]
    domain_counts: dict[str, int]
    risky_ops_found: dict[str, int]
    custom_domains: tuple[str, ...]
    has_inlined_nms: bool
    recommendation: str

    @property
    def is_ncnn_friendly(self) -> bool:
        return not self.risky_ops_found and not self.custom_domains


@dataclass(frozen=True, slots=True)
class NcnnPrecisionReport:
    """Precision and parameter audit of an NCNN `.param` (and optional `.bin`) model."""

    param_path: str
    bin_path: str | None
    layer_count: int
    weight_count: int
    bias_count: int
    has_int8_markers: bool
    detected_precision: str
    bin_size_bytes: int | None = None
    ratio_to_fp32: float | None = None
    ratio_to_fp16: float | None = None


def audit_onnx_graph(model_or_path: Any) -> OnnxAuditReport:
    """
    Inspect an ONNX ModelProto or `.onnx` file path for operators that break or
    degrade `onnx2ncnn` conversion (such as inlined `MultiClassNMS` or `GatherND`).
    """
    if isinstance(model_or_path, (str, Path)):
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "onnx is required to load .onnx files from disk. "
                "Install with: pip install 'pi5-ncnn-rtsp-vision[onnx]'"
            ) from exc
        model = onnx.load(str(model_or_path))
    else:
        model = model_or_path

    nodes = list(model.graph.node)
    op_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()

    for node in nodes:
        op_counter[str(node.op_type)] += 1
        domain = str(node.domain) if getattr(node, "domain", "") else "ai.onnx"
        domain_counter[domain] += 1

    risky_found = {
        op: count for op, count in op_counter.items() if op in NCNN_RISKY_ONNX_OPS
    }
    custom_domains = tuple(
        sorted(d for d in domain_counter if d not in ("ai.onnx", ""))
    )
    has_nms = any(
        op.startswith("NonMaxSuppression")
        or op in ("MulticlassNMS", "MultiClassNMS", "MatrixNMS")
        for op in risky_found
    )

    if has_nms:
        recommendation = (
            "Inlined NMS post-processing detected. Re-export from PaddleDetection with "
            "export.post_process=False and decode DFL + NMS on the host via PicoDetDecoder."
        )
    elif risky_found or custom_domains:
        recommendation = (
            "Non-standard or control-flow ONNX operators detected. Simplify with onnxsim "
            "or strip post-processing heads prior to onnx2ncnn."
        )
    else:
        recommendation = (
            "Graph contains standard feed-forward operators suitable for onnx2ncnn."
        )

    return OnnxAuditReport(
        total_nodes=len(nodes),
        op_counts=dict(op_counter.most_common()),
        domain_counts=dict(domain_counter),
        risky_ops_found=risky_found,
        custom_domains=custom_domains,
        has_inlined_nms=has_nms,
        recommendation=recommendation,
    )


def _parse_numeric_token(raw: str) -> int | float | str:
    try:
        if re.search(r"[.eE]", raw):
            return float(raw)
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_ncnn_param_layers(param_path: str | Path) -> list[dict[str, Any]]:
    """Parse an NCNN `.param` file into layer records `{type, name, kv}`."""
    path = Path(param_path)
    if not path.exists():
        raise FileNotFoundError(f"NCNN .param file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []

    layers: list[dict[str, Any]] = []
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        layer_type, layer_name = tokens[0], tokens[1]
        kv: dict[int, int | float | str] = {}
        for tok in tokens[2:]:
            if "=" not in tok:
                continue
            key_str, val_str = tok.split("=", 1)
            try:
                key_int = int(key_str)
            except ValueError:
                continue
            kv[key_int] = _parse_numeric_token(val_str)
        layers.append({"type": layer_type, "name": layer_name, "kv": kv})
    return layers


def inspect_ncnn_precision(
    param_path: str | Path,
    bin_path: str | Path | None = None,
) -> NcnnPrecisionReport:
    """
    Inspect an NCNN `.param` file (and optional `.bin` weights file) to determine
    whether the model uses `INT8`, `FP16`, or `FP32` precision.
    """
    layers = parse_ncnn_param_layers(param_path)
    has_int8 = False
    weight_count = 0
    bias_count = 0

    for layer in layers:
        ltype = layer["type"]
        kv = layer["kv"]
        if ltype in ("Quantize", "Dequantize", "Requantize"):
            has_int8 = True

        if ltype in ("Convolution", "ConvolutionDepthWise"):
            weight_count += int(kv.get(6, 0))
            if int(kv.get(5, 0)) == 1:
                bias_count += int(kv.get(0, 0))
            if int(kv.get(8, 0)) != 0:
                has_int8 = True
        elif ltype == "InnerProduct":
            weight_count += int(kv.get(2, 0))
            if int(kv.get(1, 0)) == 1:
                bias_count += int(kv.get(0, 0))
            if int(kv.get(8, 0)) != 0:
                has_int8 = True

    bin_size: int | None = None
    ratio_fp32: float | None = None
    ratio_fp16: float | None = None

    if bin_path is not None:
        bpath = Path(bin_path)
        if not bpath.exists():
            raise FileNotFoundError(f"NCNN .bin file not found: {bpath}")
        bin_size = os.path.getsize(bpath)
        bias_bytes = bias_count * 4
        expected_fp32 = weight_count * 4 + bias_bytes
        expected_fp16 = weight_count * 2 + bias_bytes
        if expected_fp32 > 0:
            ratio_fp32 = bin_size / expected_fp32
        if expected_fp16 > 0:
            ratio_fp16 = bin_size / expected_fp16

    if has_int8:
        detected = "INT8"
    elif ratio_fp16 is not None and ratio_fp32 is not None:
        detected = (
            "FP16"
            if abs(ratio_fp16 - 1.0) + 1e-6 < abs(ratio_fp32 - 1.0)
            else "FP32"
        )
    else:
        detected = "FP32/FP16 (Topology Only)"

    return NcnnPrecisionReport(
        param_path=str(param_path),
        bin_path=str(bin_path) if bin_path is not None else None,
        layer_count=len(layers),
        weight_count=weight_count,
        bias_count=bias_count,
        has_int8_markers=has_int8,
        detected_precision=detected,
        bin_size_bytes=bin_size,
        ratio_to_fp32=ratio_fp32,
        ratio_to_fp16=ratio_fp16,
    )
