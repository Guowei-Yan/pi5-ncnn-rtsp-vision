from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from pi5_ncnn_vision.audit import audit_onnx_graph, inspect_ncnn_precision
from pi5_ncnn_vision.engine import (
    HybridModelRouter,
    NCNNDetector,
    VulkanRuntimeContext,
)
from pi5_ncnn_vision.ocr import CropOCRProcessor
from pi5_ncnn_vision.stream import MultiStreamPipeline
from pi5_ncnn_vision.types import DecoderSpec, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi5-vision",
        description=(
            "Raspberry Pi CM5 / Pi 5 NCNN (Cortex-A76 INT8 + VideoCore VII Vulkan FP16) "
            "multi-stream RTSP detection, DFL decoding, and two-stage OCR toolkit."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_onnx = sub.add_parser(
        "audit-onnx",
        help="Audit an ONNX model for NCNN-incompatible post-processing operators.",
    )
    p_onnx.add_argument("onnx_path", type=Path, help="Path to .onnx model file.")

    p_ncnn = sub.add_parser(
        "inspect-ncnn",
        help="Inspect NCNN .param (and optional .bin) for INT8/FP16/FP32 precision.",
    )
    p_ncnn.add_argument("--param", required=True, type=Path, help="Path to .param file.")
    p_ncnn.add_argument("--bin", default=None, type=Path, help="Optional path to .bin file.")

    p_bench = sub.add_parser(
        "benchmark",
        help="Benchmark Cortex-A76 CPU INT8 vs VideoCore VII Vulkan FP16 NCNN latency.",
    )
    p_bench.add_argument("--param-cpu", required=True, type=Path)
    p_bench.add_argument("--bin-cpu", required=True, type=Path)
    p_bench.add_argument("--param-gpu", default=None, type=Path)
    p_bench.add_argument("--bin-gpu", default=None, type=Path)
    p_bench.add_argument("--runs", type=int, default=10)
    p_bench.add_argument("--threads", type=int, default=2)
    p_bench.add_argument("--gpu-device", type=int, default=0)
    p_bench.add_argument("--input-width", type=int, default=320)
    p_bench.add_argument("--input-height", type=int, default=320)
    p_bench.add_argument("--num-classes", type=int, default=80)

    p_run = sub.add_parser(
        "run-rtsp",
        help="Run the multi-stream zero-buffer RTSP NCNN detection pipeline.",
    )
    p_run.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    p_run.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run duration in seconds (default: run until interrupted).",
    )

    return parser


def _cmd_audit_onnx(args: argparse.Namespace) -> int:
    report = audit_onnx_graph(args.onnx_path)
    print(f"ONNX Model      : {args.onnx_path}")
    print(f"Total Nodes     : {report.total_nodes}")
    print(f"NCNN Friendly   : {report.is_ncnn_friendly}")
    print(f"Inlined NMS     : {report.has_inlined_nms}")
    if report.risky_ops_found:
        print("Risky Operators :")
        for op, cnt in report.risky_ops_found.items():
            print(f"  - {op}: {cnt}")
    print(f"Recommendation  : {report.recommendation}")
    return 0


def _cmd_inspect_ncnn(args: argparse.Namespace) -> int:
    report = inspect_ncnn_precision(args.param, args.bin)
    print(f"Param File      : {report.param_path}")
    if report.bin_path:
        print(f"Bin File        : {report.bin_path} ({report.bin_size_bytes} bytes)")
    print(f"Parsed Layers   : {report.layer_count}")
    print(f"Weights / Biases: {report.weight_count} / {report.bias_count}")
    print(f"INT8 Markers    : {report.has_int8_markers}")
    print(f"Precision       : {report.detected_precision}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    param_gpu = args.param_gpu or args.param_cpu
    bin_gpu = args.bin_gpu or args.bin_cpu
    spec = DecoderSpec(
        input_width=args.input_width,
        input_height=args.input_height,
        num_classes=args.num_classes,
    )

    with VulkanRuntimeContext(enabled=True) as vk:
        if vk.gpu_count == 0:
            print("No Vulkan-capable GPU detected; running CPU-only detector check.")
            with NCNNDetector(
                args.param_cpu,
                args.bin_cpu,
                decoder_spec=spec,
                num_threads=args.threads,
                use_vulkan=False,
            ) as cpu_det:
                print(f"Loaded {cpu_det.backend_name} successfully.")
            return 0

        with (
            NCNNDetector(
                args.param_cpu,
                args.bin_cpu,
                decoder_spec=spec,
                num_threads=args.threads,
                use_vulkan=False,
            ) as cpu_det,
            NCNNDetector(
                param_gpu,
                bin_gpu,
                decoder_spec=spec,
                num_threads=args.threads,
                use_vulkan=True,
                gpu_device=args.gpu_device,
            ) as gpu_det,
            HybridModelRouter(cpu_det, gpu_det, gpu_stream_id=0) as router,
        ):
            report = router.benchmark(runs=args.runs)
            if report is not None:
                print(
                    f"CPU ({cpu_det.backend_name}) : "
                    f"{report.cpu_mean_ms:.2f} ms +/- {report.cpu_std_ms:.2f} ms"
                )
                print(
                    f"GPU ({gpu_det.backend_name}) : "
                    f"{report.gpu_mean_ms:.2f} ms +/- {report.gpu_std_ms:.2f} ms"
                )
                print(f"Vulkan Speedup             : {report.speedup:.2f}x")
    return 0


def _cmd_run_rtsp(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.from_yaml(args.config)
    with ExitStack() as stack:
        vk = stack.enter_context(VulkanRuntimeContext(enabled=cfg.use_hybrid_mode))
        use_hybrid = cfg.use_hybrid_mode and vk.gpu_count > 0
        gpu_stream = (
            cfg.gpu_stream_id
            if cfg.gpu_stream_id is not None
            else random.randint(0, len(cfg.rtsp_urls) - 1)
        )

        cpu_det = stack.enter_context(
            NCNNDetector(
                cfg.model_param_cpu,
                cfg.model_bin_cpu,
                decoder_spec=cfg.decoder,
                num_threads=cfg.ncnn_num_threads,
                use_vulkan=False,
            )
        )
        gpu_det = (
            stack.enter_context(
                NCNNDetector(
                    cfg.model_param_gpu or cfg.model_param_cpu,
                    cfg.model_bin_gpu or cfg.model_bin_cpu,
                    decoder_spec=cfg.decoder,
                    num_threads=cfg.ncnn_num_threads,
                    use_vulkan=True,
                    gpu_device=cfg.gpu_device_id,
                )
            )
            if use_hybrid
            else None
        )
        router = stack.enter_context(
            HybridModelRouter(cpu_det, gpu_det, gpu_stream_id=gpu_stream)
        )
        if use_hybrid and cfg.gpu_validation_runs > 0:
            router.benchmark(runs=cfg.gpu_validation_runs)

        ocr_proc = (
            CropOCRProcessor(
                reco_arch=cfg.ocr_reco_arch,
                target_class_id=cfg.sign_class_id,
                min_det_score=cfg.sign_score_threshold,
                min_ocr_confidence=cfg.ocr_min_confidence,
                max_crops_per_frame=cfg.ocr_max_crops_per_frame,
                allowed_speeds=cfg.allowed_speeds,
            )
            if cfg.enable_sign_ocr
            else None
        )

        pipeline = MultiStreamPipeline(cfg, router, ocr_processor=ocr_proc)
        summary = pipeline.run(max_duration_sec=args.duration)
        print(json.dumps(summary, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "audit-onnx":
        return _cmd_audit_onnx(args)
    if args.command == "inspect-ncnn":
        return _cmd_inspect_ncnn(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "run-rtsp":
        return _cmd_run_rtsp(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
