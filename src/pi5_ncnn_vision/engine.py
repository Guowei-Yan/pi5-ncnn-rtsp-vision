from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pi5_ncnn_vision.decoder import PicoDetDecoder
from pi5_ncnn_vision.types import DecoderSpec, Detection

IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _require_ncnn() -> Any:
    try:
        import ncnn
        return ncnn
    except ImportError as exc:
        raise RuntimeError(
            "python-ncnn is required for NCNN inference. "
            "Install with: pip install ncnn"
        ) from exc


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Latency comparison between Cortex-A76 CPU INT8 and VideoCore VII Vulkan FP16."""

    runs: int
    cpu_mean_ms: float
    cpu_std_ms: float
    gpu_mean_ms: float
    gpu_std_ms: float
    speedup: float


class VulkanRuntimeContext:
    """Context manager for safe NCNN Vulkan GPU allocator lifecycle management."""

    def __init__(self, enabled: bool = True, ncnn_module: Any | None = None) -> None:
        self._enabled = enabled
        self._ncnn = ncnn_module
        self.gpu_count: int = 0
        self._initialized: bool = False

    def __enter__(self) -> VulkanRuntimeContext:
        if not self._enabled:
            return self
        ncnn = self._ncnn or _require_ncnn()
        self._ncnn = ncnn
        gpu_count = int(ncnn.get_gpu_count())
        if gpu_count > 0:
            ncnn.create_gpu_instance()
            self.gpu_count = gpu_count
            self._initialized = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        if self._initialized and self._ncnn is not None:
            try:
                self._ncnn.destroy_gpu_instance()
            finally:
                self._initialized = False


class NCNNDetector:
    """
    Asymmetric NCNN PicoDet inference engine for Raspberry Pi CM5 / Pi 5.

    - On Cortex-A76 CPU (`use_vulkan=False`): enables `INT8` storage, packed layout,
      and `INT8` arithmetic.
    - On VideoCore VII Vulkan GPU (`use_vulkan=True`): enables `FP16` storage, packed
      layout, `FP16` arithmetic, `shader_pack8`, and `image_storage` (avoiding slow
      Vulkan INT8 shader paths on VideoCore VII).
    """

    OUTPUT_BLOB_NAMES: tuple[str, ...] = (
        "out0",
        "out1",
        "out2",
        "out3",
        "out4",
        "out5",
        "out6",
        "out7",
    )

    def __init__(
        self,
        param_path: str | Path,
        bin_path: str | Path,
        *,
        decoder_spec: DecoderSpec | None = None,
        num_threads: int = 2,
        use_vulkan: bool = False,
        gpu_device: int = 0,
        ncnn_module: Any | None = None,
    ) -> None:
        self._ncnn = ncnn_module or _require_ncnn()
        self.param_path = Path(param_path)
        self.bin_path = Path(bin_path)
        if not self.param_path.exists():
            raise FileNotFoundError(f"NCNN .param file not found: {self.param_path}")
        if not self.bin_path.exists():
            raise FileNotFoundError(f"NCNN .bin file not found: {self.bin_path}")

        self.decoder = PicoDetDecoder(decoder_spec)
        self.use_vulkan = bool(use_vulkan)
        self.gpu_device = int(gpu_device)
        self.num_threads = int(num_threads)

        self._mean_vals = (IMAGENET_MEAN_RGB * 255.0).tolist()
        self._norm_vals = (1.0 / (IMAGENET_STD_RGB * 255.0)).tolist()

        self._net = self._ncnn.Net()
        self._configure_options()

        ret_param = self._net.load_param(str(self.param_path))
        ret_bin = self._net.load_model(str(self.bin_path))
        if ret_param != 0 or ret_bin != 0:
            self.close()
            raise RuntimeError(
                f"Failed to load NCNN model (param_ret={ret_param}, bin_ret={ret_bin})."
            )

    def _configure_options(self) -> None:
        opt = self._net.opt
        opt.num_threads = self.num_threads
        opt.use_packing_layout = True
        opt.use_winograd_convolution = True

        if not self.use_vulkan:
            opt.use_int8_storage = True
            opt.use_int8_arithmetic = True
            opt.use_int8_packed = True
        else:
            opt.use_int8_storage = True
            opt.use_fp16_packed = True
            opt.use_fp16_storage = True
            opt.use_fp16_arithmetic = True
            opt.use_shader_pack8 = True
            opt.use_image_storage = True
            self._net.set_vulkan_device(self.gpu_device)
            opt.use_vulkan_compute = True

    @property
    def backend_name(self) -> str:
        return f"GPU{self.gpu_device}/Vulkan" if self.use_vulkan else "CPU/INT8"

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[Any, tuple[int, int]]:
        """Convert HxWx3 BGR uint8 image to resized, ImageNet-normalized `ncnn.Mat`."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape {frame_bgr.shape}.")
        orig_h, orig_w = frame_bgr.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError("Input frame has zero width or height.")

        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        mat = self._ncnn.Mat.from_pixels_resize(
            rgb,
            self._ncnn.Mat.PixelType.PIXEL_RGB,
            orig_w,
            orig_h,
            self.decoder.spec.input_width,
            self.decoder.spec.input_height,
        )
        mat.substract_mean_normalize(self._mean_vals, self._norm_vals)
        return mat, (orig_h, orig_w)

    def forward(self, mat_in: Any) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Execute NCNN graph extraction and pair multi-scale classification & DFL blobs."""
        if self._net is None:
            raise RuntimeError("NCNNDetector has already been closed.")

        ex = self._net.create_extractor()
        try:
            ex.set_num_threads(self.num_threads)
            if self.use_vulkan:
                ex.set_vulkan_compute(True)
        except AttributeError:
            pass

        ex.input("in0", mat_in)
        raw_blobs: dict[str, np.ndarray] = {}
        for name in self.OUTPUT_BLOB_NAMES:
            out_mat = self._ncnn.Mat()
            ex.extract(name, out_mat)
            raw_blobs[name] = np.array(out_mat, copy=True)

        return self.decoder.pair_multiscale_blobs(raw_blobs)

    def postprocess(
        self,
        paired_levels: Sequence[tuple[int, np.ndarray, np.ndarray]],
        orig_size: tuple[int, int],
    ) -> list[Detection]:
        return self.decoder.decode(paired_levels, orig_size)

    def detect_timed(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[Detection], float, float, float]:
        """Run full preprocess -> forward -> postprocess and return (detections, pre_ms, fwd_ms, post_ms)."""
        t0 = time.perf_counter()
        mat_in, orig_size = self.preprocess(frame_bgr)
        t1 = time.perf_counter()
        paired = self.forward(mat_in)
        t2 = time.perf_counter()
        detections = self.postprocess(paired, orig_size)
        t3 = time.perf_counter()
        return (
            detections,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        detections, _, _, _ = self.detect_timed(frame_bgr)
        return detections

    def close(self) -> None:
        if getattr(self, "_net", None) is not None:
            self._net.clear()
            self._net = None

    def __enter__(self) -> NCNNDetector:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


class HybridModelRouter:
    """
    Routes multi-stream inference across Cortex-A76 CPU (`INT8`) and VideoCore VII
    Vulkan GPU (`FP16`) detectors with deterministic lifecycle cleanup.
    """

    def __init__(
        self,
        cpu_detector: NCNNDetector,
        gpu_detector: NCNNDetector | None = None,
        gpu_stream_id: int = 0,
    ) -> None:
        self.cpu_detector = cpu_detector
        self.gpu_detector = gpu_detector
        self.gpu_stream_id = gpu_stream_id if gpu_detector is not None else -1

    def select_detector(self, stream_id: int) -> NCNNDetector:
        if self.gpu_detector is not None and stream_id == self.gpu_stream_id:
            return self.gpu_detector
        return self.cpu_detector

    def backend_for_stream(self, stream_id: int) -> str:
        if self.gpu_detector is not None and stream_id == self.gpu_stream_id:
            return "GPU"
        return "CPU"

    def benchmark(
        self,
        runs: int = 5,
        warmup: int = 2,
        frame_bgr: np.ndarray | None = None,
    ) -> BenchmarkReport | None:
        if self.gpu_detector is None or runs <= 0:
            return None

        sample = (
            frame_bgr
            if frame_bgr is not None
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )

        for _ in range(max(0, warmup)):
            mat_c, _ = self.cpu_detector.preprocess(sample)
            self.cpu_detector.forward(mat_c)
            mat_g, _ = self.gpu_detector.preprocess(sample)
            self.gpu_detector.forward(mat_g)

        cpu_times: list[float] = []
        for _ in range(runs):
            mat_in, _ = self.cpu_detector.preprocess(sample)
            t0 = time.perf_counter()
            self.cpu_detector.forward(mat_in)
            cpu_times.append((time.perf_counter() - t0) * 1000.0)

        gpu_times: list[float] = []
        for _ in range(runs):
            mat_in, _ = self.gpu_detector.preprocess(sample)
            t0 = time.perf_counter()
            self.gpu_detector.forward(mat_in)
            gpu_times.append((time.perf_counter() - t0) * 1000.0)

        cpu_mean = float(np.mean(cpu_times))
        cpu_std = float(np.std(cpu_times))
        gpu_mean = float(np.mean(gpu_times))
        gpu_std = float(np.std(gpu_times))
        speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0.0

        return BenchmarkReport(
            runs=runs,
            cpu_mean_ms=cpu_mean,
            cpu_std_ms=cpu_std,
            gpu_mean_ms=gpu_mean,
            gpu_std_ms=gpu_std,
            speedup=speedup,
        )

    def close(self) -> None:
        if self.gpu_detector is not None:
            self.gpu_detector.close()
        self.cpu_detector.close()

    def __enter__(self) -> HybridModelRouter:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
