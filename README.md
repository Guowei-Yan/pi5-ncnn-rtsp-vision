# pi5-ncnn-rtsp-vision

Zero-buffer multi-stream RTSP object detection and two-stage sign OCR for **Raspberry Pi Compute Module 5 (CM5) / Raspberry Pi 5** (Broadcom BCM2712), built on **NCNN** with asymmetric **Cortex-A76 CPU `INT8` + VideoCore VII Vulkan GPU `FP16`** execution and vectorized NumPy **Distribution Focal Loss (DFL)** decoding.

---

## Why This Exists

Deploying multi-camera RTSP vision on an embedded ARM board without a dedicated PCIe NPU exposes four practical bottlenecks:

1. **Transformer/DETR CPU saturation**: End-to-end detectors such as `D-FINE` and `RT-DETR` deliver strong accuracy on desktop GPUs, but quickly saturate quad-core Cortex-A76 CPUs when decoding and inferring across 4 concurrent RTSP streams alongside downstream OCR.
2. **`onnx2ncnn` failures on inlined NMS**: Exporting `PicoDet` (`LCNet` backbone + `LCPAN` neck + `PicoHeadV2` head) with `export.post_process=True` embeds `MultiClassNMS`, `GatherND`, and dynamic control-flow operators into the ONNX graph. These operators either fail during `onnx2ncnn` conversion or break `ncnn2int8` post-training quantization.
3. **VideoCore VII Vulkan `INT8` vs `FP16` asymmetry**: On the Raspberry Pi 5 / CM5 BCM2712 SoC, the Cortex-A76 CPU cores accelerate `INT8` arithmetic via ARM NEON dot-product instructions, whereas the integrated **VideoCore VII GPU** executes `FP16` Vulkan compute shaders (`shader_pack8` + `image_storage`) substantially faster than `INT8` Vulkan paths.
4. **RTSP buffer drift**: OpenCV's default FFmpeg capture queue buffers incoming H.264/H.265 packets. Whenever inference runs at a lower rate than the camera stream (e.g., 2–5 FPS sampling against a 25 FPS stream), buffered frames accumulate and detections lag seconds behind real time.

---

## Architecture

```
RTSP Cam 0..N ──► [FFmpeg nobuffer + CAP_PROP_BUFFERSIZE=1]
                         │
                         ▼
                 [FrameSlot 0..N]  (Lock-protected 1-slot overwrite mailbox; drop-oldest)
                         │
                         ▼
               [HybridModelRouter]
                 ├─► Stream K     ──► VideoCore VII Vulkan GPU (FP16 storage/arithmetic + shader_pack8)
                 └─► Other Streams ──► Cortex-A76 CPU (INT8 storage/arithmetic + packed layout)
                         │
                         ▼
                [PicoDetDecoder]   (4-stride DFL softmax integral projection + argpartition + hard NMS)
                         │
                         ├─► Optional [CropOCRProcessor] (In-memory sign crop ──► docTR CRNN recognition)
                         ▼
              [RotatingNDJSONSink] (Batch-buffered, size-rotated .ndjson telemetry)
```

### 1. Stripped-Postprocess `PicoHeadV2` & Host-Side DFL Decoding (`pi5_ncnn_vision.decoder`)

When exported with `export.post_process=False`, the NCNN graph terminates at 8 raw head tensors across FPN strides $s \in \{8, 16, 32, 64\}$:

- **`out0..out3` (Classification + Centerness)**: Shape $(L_s, C)$, where $L_s = (H / s) \times (W / s)$ (`1600, 400, 100, 25` for $320 \times 320$ input; `6400, 1600, 400, 100` for $640 \times 640$ input).
- **`out4..out7` (DFL Bounding-Box Regression)**: Shape $(L_s, 4 \times (\text{reg\_max} + 1)) = (L_s, 32)$ for $\text{reg\_max} = 7$.

For each anchor surviving `score_threshold`, `PicoDetDecoder` projects the discrete distribution over bins $\{0, 1, \dots, 7\}$ into scalar distances $(l, t, r, b)$ from the grid cell center $(c_x, c_y) = ((x + 0.5)s, (y + 0.5)s)$:

$$\hat{d} = s \cdot \sum_{k=0}^{\text{reg\_max}} k \cdot \text{softmax}(z)_k$$

$$x_1 = c_x - \hat{d}_l, \quad y_1 = c_y - \hat{d}_t, \quad x_2 = c_x + \hat{d}_r, \quad y_2 = c_y + \hat{d}_b$$

Filtering by `score_threshold` *before* running the softmax integral projection restricts matrix multiplication to high-confidence anchors, keeping host-side decoding under `0.5 ms` per frame in pure NumPy.

### 2. Asymmetric CPU `INT8` + Vulkan `FP16` Execution (`pi5_ncnn_vision.engine`)

`NCNNDetector` configures `ncnn.Net.opt` specifically for each BCM2712 compute unit:

| Option | Cortex-A76 CPU (`use_vulkan=False`) | VideoCore VII GPU (`use_vulkan=True`) |
| :--- | :--- | :--- |
| `use_int8_storage` | `True` | `True` |
| `use_int8_arithmetic` | `True` | `False` |
| `use_int8_packed` | `True` | `False` |
| `use_fp16_storage` / `packed` | `False` | `True` |
| `use_fp16_arithmetic` | `False` | `True` |
| `use_shader_pack8` | `False` | `True` |
| `use_image_storage` | `False` | `True` |

### 3. Zero-Buffer RTSP & Single-Slot `FrameSlot` (`pi5_ncnn_vision.stream`)

Each RTSP reader thread opens its stream with:

```text
rtsp_transport;tcp|fflags;nobuffer|probesize;32|analyzeduration;0|stimeout;5000000
```

and writes sampled frames into a dedicated `FrameSlot`. If the inference worker is busy when a newer frame arrives, `FrameSlot.put()` overwrites the unconsumed frame in place and increments `overwrite_drops`, guaranteeing bounded end-to-end latency.

---

## Model Export & NCNN Quantization Recipe

1. **Export from PaddleDetection without post-processing**:
   ```bash
   python tools/export_model.py \
     -c configs/picodet/picodet_m_320_coco_lcnet.yml \
     -o "weights=picodet_m_320_coco_lcnet.pdparams TestReader.inputs_def.image_shape=[1,3,320,320] export.post_process=False export.nms=False use_gpu=False" \
     --output_dir inference_model
   ```

2. **Convert to ONNX and simplify**:
   ```bash
   paddle2onnx \
     --model_dir inference_model/picodet_m_320_coco_lcnet \
     --model_filename model.pdmodel \
     --params_filename model.pdiparams \
     --opset_version 11 \
     --save_file picodet_m_320_no_postprocess.onnx

   python -m onnxsim picodet_m_320_no_postprocess.onnx picodet_m_320_sim.onnx
   ```

3. **Audit ONNX compatibility before NCNN conversion**:
   ```bash
   pi5-vision audit-onnx picodet_m_320_sim.onnx
   ```

4. **Convert to NCNN, calibrate `INT8`, and optimize**:
   ```bash
   onnx2ncnn picodet_m_320_sim.onnx picodet_m_320.param picodet_m_320.bin
   ncnnoptimize picodet_m_320.param picodet_m_320.bin picodet_m_320_opt.param picodet_m_320_opt.bin 0

   # Generate calibration table and quantize to INT8 for Cortex-A76 CPU
   ncnn2table picodet_m_320_opt.param picodet_m_320_opt.bin imagelist.txt picodet_m_320.table mean=[123.675,116.28,103.53] norm=[0.017125,0.017507,0.017429] shape=[320,320,3] pixel=RGB thread=4 method=kl
   ncnn2int8 picodet_m_320_opt.param picodet_m_320_opt.bin picodet_m_320_quant.param picodet_m_320_quant.bin picodet_m_320.table
   ncnnoptimize picodet_m_320_quant.param picodet_m_320_quant.bin picodet_m_320_quant_opt.param picodet_m_320_quant_opt.bin 0
   ```

   Reference NCNN graph topology files (`models/picodet_m_320_quant_opt.param` for 80-class COCO $320 \times 320$ and `models/picodet_s_640_two_class_int8.param` for 2-class `person` + `speed_limit_sign` $640 \times 640$) are included in [`models/`](models/).

---

## Installation & Usage

```bash
pip install -e ".[runtime,ocr,onnx,dev]"
```

### Inspect NCNN Model Precision

```bash
pi5-vision inspect-ncnn --param models/picodet_m_320_quant_opt.param
```

### Benchmark Cortex-A76 CPU `INT8` vs VideoCore VII Vulkan `FP16`

```bash
pi5-vision benchmark \
  --param-cpu models/picodet_m_320_quant_opt.param \
  --bin-cpu models/picodet_m_320_quant_opt.bin \
  --runs 20 --threads 2
```

### Run Multi-Stream RTSP Pipeline

Edit [`configs/multi_stream.example.yaml`](configs/multi_stream.example.yaml) with your RTSP endpoints and model `.bin` paths, then run:

```bash
pi5-vision run-rtsp --config configs/multi_stream.example.yaml
```

### Run Unit Tests

```bash
pytest -v
```

---

## License

MIT
