from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pi5_ncnn_vision.audit import audit_onnx_graph, inspect_ncnn_precision
from pi5_ncnn_vision.cli import main as cli_main
from pi5_ncnn_vision.engine import (
    HybridModelRouter,
    NCNNDetector,
    VulkanRuntimeContext,
)
from pi5_ncnn_vision.types import DecoderSpec

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_inspect_real_picodet_ncnn_param_files() -> None:
    m320 = MODELS_DIR / "picodet_m_320_quant_opt.param"
    m320_fp = MODELS_DIR / "picodet_m_320_no_postprocess_opt.param"
    s640 = MODELS_DIR / "picodet_s_640_two_class_int8.param"

    rep_m = inspect_ncnn_precision(m320)
    assert rep_m.layer_count == 177
    assert rep_m.has_int8_markers is True
    assert rep_m.detected_precision == "INT8"
    assert rep_m.weight_count > 3_000_000

    rep_fp = inspect_ncnn_precision(m320_fp)
    assert rep_fp.has_int8_markers is False
    assert rep_fp.detected_precision == "FP32/FP16 (Topology Only)"

    rep_s = inspect_ncnn_precision(s640)
    assert rep_s.has_int8_markers is True
    assert rep_s.detected_precision == "INT8"
    assert rep_s.weight_count > 1_000_000


def test_inspect_synthetic_fp16_and_fp32_models(tmp_path: Path) -> None:
    param_file = tmp_path / "conv_fp.param"
    param_file.write_text(
        "7767517\n"
        "2 2\n"
        "Input in0 0 1 in0\n"
        "Convolution conv1 1 1 in0 out0 0=16 1=3 5=1 6=1000\n",
        encoding="utf-8",
    )
    # 1000 weights + 16 biases:
    # FP16 = 1000 * 2 + 16 * 4 = 2064 bytes
    # FP32 = 1000 * 4 + 16 * 4 = 4064 bytes
    bin_fp16 = tmp_path / "conv_fp16.bin"
    bin_fp16.write_bytes(b"\x00" * 2064)
    rep16 = inspect_ncnn_precision(param_file, bin_fp16)
    assert rep16.detected_precision == "FP16"

    bin_fp32 = tmp_path / "conv_fp32.bin"
    bin_fp32.write_bytes(b"\x00" * 4064)
    rep32 = inspect_ncnn_precision(param_file, bin_fp32)
    assert rep32.detected_precision == "FP32"


def test_audit_onnx_graph_detects_inlined_nms() -> None:
    inlined_model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                SimpleNamespace(op_type="Conv", domain=""),
                SimpleNamespace(op_type="Softmax", domain=""),
                SimpleNamespace(op_type="MultiClassNMS", domain="paddle"),
                SimpleNamespace(op_type="GatherND", domain=""),
            ]
        )
    )
    report = audit_onnx_graph(inlined_model)
    assert report.is_ncnn_friendly is False
    assert report.has_inlined_nms is True
    assert "MultiClassNMS" in report.risky_ops_found
    assert "GatherND" in report.risky_ops_found
    assert "paddle" in report.custom_domains
    assert "export.post_process=False" in report.recommendation

    clean_model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                SimpleNamespace(op_type="Conv", domain=""),
                SimpleNamespace(op_type="HardSwish", domain=""),
                SimpleNamespace(op_type="Resize", domain=""),
            ]
        )
    )
    clean_report = audit_onnx_graph(clean_model)
    assert clean_report.is_ncnn_friendly is True
    assert clean_report.has_inlined_nms is False


def test_ncnn_detector_asymmetric_options_and_lifecycle(tmp_path: Path) -> None:
    param_path = tmp_path / "dummy.param"
    bin_path = tmp_path / "dummy.bin"
    param_path.write_text("7767517\n0 0\n", encoding="utf-8")
    bin_path.write_bytes(b"\x00" * 16)

    created_nets: list[FakeNet] = []
    vulkan_events: list[str] = []

    class FakeMat:
        class PixelType:
            PIXEL_RGB = 1

        def __init__(self, arr: np.ndarray | None = None) -> None:
            self._arr = arr if arr is not None else np.zeros((1, 1), dtype=np.float32)

        @classmethod
        def from_pixels_resize(
            cls, rgb: np.ndarray, pixel_type: int, w0: int, h0: int, w1: int, h1: int
        ) -> FakeMat:
            return cls(np.zeros((h1, w1, 3), dtype=np.float32))

        def substract_mean_normalize(
            self, mean_vals: list[float], norm_vals: list[float]
        ) -> None:
            self.normalized = (mean_vals, norm_vals)

        def __array__(
            self, dtype: np.dtype | None = None, copy: bool | None = None
        ) -> np.ndarray:
            arr = self._arr.astype(dtype) if dtype else self._arr
            return arr.copy() if copy else arr

    class FakeExtractor:
        def __init__(self) -> None:
            self.blobs = {
                "out0": np.zeros((1600, 80), dtype=np.float32),
                "out1": np.zeros((400, 80), dtype=np.float32),
                "out2": np.zeros((100, 80), dtype=np.float32),
                "out3": np.zeros((25, 80), dtype=np.float32),
                "out4": np.zeros((1600, 32), dtype=np.float32),
                "out5": np.zeros((400, 32), dtype=np.float32),
                "out6": np.zeros((100, 32), dtype=np.float32),
                "out7": np.zeros((25, 32), dtype=np.float32),
            }
            self.blobs["out2"][12, 0] = 0.88

        def set_num_threads(self, n: int) -> None:
            pass

        def set_vulkan_compute(self, flag: bool) -> None:
            pass

        def input(self, name: str, mat: FakeMat) -> None:
            assert name == "in0"

        def extract(self, name: str, out_mat: FakeMat) -> None:
            out_mat._arr = self.blobs[name]

    class FakeNet:
        def __init__(self) -> None:
            self.opt = SimpleNamespace()
            self.vulkan_device: int | None = None
            self.cleared = False
            created_nets.append(self)

        def set_vulkan_device(self, dev: int) -> None:
            self.vulkan_device = dev

        def load_param(self, path: str) -> int:
            return 0

        def load_model(self, path: str) -> int:
            return 0

        def create_extractor(self) -> FakeExtractor:
            return FakeExtractor()

        def clear(self) -> None:
            self.cleared = True

    fake_ncnn = SimpleNamespace(
        Net=FakeNet,
        Mat=FakeMat,
        get_gpu_count=lambda: 1,
        create_gpu_instance=lambda: vulkan_events.append("created"),
        destroy_gpu_instance=lambda: vulkan_events.append("destroyed"),
    )

    with VulkanRuntimeContext(enabled=True, ncnn_module=fake_ncnn) as vk:
        assert vk.gpu_count == 1
        with (
            NCNNDetector(
                param_path,
                bin_path,
                decoder_spec=DecoderSpec(),
                num_threads=2,
                use_vulkan=False,
                ncnn_module=fake_ncnn,
            ) as cpu_det,
            NCNNDetector(
                param_path,
                bin_path,
                decoder_spec=DecoderSpec(),
                num_threads=2,
                use_vulkan=True,
                gpu_device=0,
                ncnn_module=fake_ncnn,
            ) as gpu_det,
            HybridModelRouter(cpu_det, gpu_det, gpu_stream_id=0) as router,
        ):
            # Verify Cortex-A76 CPU INT8 vs VideoCore VII Vulkan FP16 option flags
            assert created_nets[0].opt.use_int8_arithmetic is True
            assert created_nets[1].opt.use_fp16_arithmetic is True
            assert created_nets[1].opt.use_shader_pack8 is True
            assert created_nets[1].opt.use_vulkan_compute is True

            sample = np.zeros((320, 320, 3), dtype=np.uint8)
            dets = router.select_detector(0).detect(sample)
            assert len(dets) == 1
            assert dets[0].class_name == "person"

            bench = router.benchmark(runs=2, warmup=1, frame_bgr=sample)
            assert bench is not None
            assert bench.runs == 2

    assert created_nets[0].cleared is True
    assert created_nets[1].cleared is True
    assert vulkan_events == ["created", "destroyed"]


def test_cli_inspect_ncnn_command(capsys: object) -> None:
    m320 = MODELS_DIR / "picodet_m_320_quant_opt.param"
    rc = cli_main(["inspect-ncnn", "--param", str(m320)])
    assert rc == 0
