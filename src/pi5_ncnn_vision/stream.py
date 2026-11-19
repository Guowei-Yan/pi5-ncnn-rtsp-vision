from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pi5_ncnn_vision.engine import HybridModelRouter
from pi5_ncnn_vision.ocr import CropOCRProcessor
from pi5_ncnn_vision.types import (
    DEFAULT_FFMPEG_CAPTURE_OPTIONS,
    Detection,
    FramePacket,
    InferenceRecord,
    PipelineConfig,
)

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


logger = logging.getLogger(__name__)


def dumps_json_line(payload: Mapping[str, Any]) -> str:
    """Serialize a mapping to a compact single-line JSON string."""
    if _orjson is not None:
        return _orjson.dumps(payload).decode("utf-8")
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class VideoCaptureProtocol(Protocol):
    """Minimal protocol matching `cv2.VideoCapture` for testable RTSP ingestion."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[str], VideoCaptureProtocol]


def configure_ffmpeg_capture_env(
    options: str = DEFAULT_FFMPEG_CAPTURE_OPTIONS,
    *,
    overwrite: bool = False,
) -> str:
    """Set `OPENCV_FFMPEG_CAPTURE_OPTIONS` explicitly when starting RTSP capture."""
    if overwrite or "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    return os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]


def default_opencv_capture_factory(url: str) -> VideoCaptureProtocol:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless (or opencv-python) is required for RTSP capture."
        ) from exc
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


class FrameSlot:
    """
    Lock-protected single-slot overwrite mailbox for a single camera stream.

    Guarantees drop-oldest semantics: if the decoder thread produces a new sampled
    frame before the inference worker consumes the previous one, the stale frame is
    overwritten in place so inference latency never accumulates behind camera buffers.
    """

    def __init__(
        self,
        stream_id: int = 0,
        *,
        notify_event: threading.Event | None = None,
    ) -> None:
        self.stream_id = stream_id
        self._lock = threading.Lock()
        self._packet: FramePacket | None = None
        self._consumed: bool = True
        self._overwrite_drops: int = 0
        self._notify_event = notify_event

    def set_notify_event(self, event: threading.Event | None) -> None:
        with self._lock:
            self._notify_event = event

    def put(
        self,
        frame_bgr: np.ndarray,
        capture_timestamp: float,
        frame_count: int,
        stream_id: int | None = None,
        *,
        capture_monotonic: float | None = None,
    ) -> None:
        resolved_id = self.stream_id if stream_id is None else stream_id
        packet = FramePacket(
            stream_id=resolved_id,
            frame_count=frame_count,
            capture_timestamp=capture_timestamp,
            frame_bgr=frame_bgr,
            capture_monotonic=capture_monotonic,
        )
        with self._lock:
            if not self._consumed and self._packet is not None:
                self._overwrite_drops += 1
            self._packet = packet
            self._consumed = False
            notify = self._notify_event
        if notify is not None:
            notify.set()

    def take_if_new(self) -> FramePacket | None:
        with self._lock:
            if self._consumed or self._packet is None:
                return None
            packet = self._packet
            self._consumed = True
            return packet

    @property
    def overwrite_drops(self) -> int:
        with self._lock:
            return self._overwrite_drops

    def peek_status(self) -> dict[str, Any] | None:
        with self._lock:
            if self._packet is None:
                return None
            return {
                "stream_id": self._packet.stream_id,
                "frame_count": self._packet.frame_count,
                "capture_timestamp": self._packet.capture_timestamp,
                "consumed": self._consumed,
                "overwrite_drops": self._overwrite_drops,
                "frame_shape": self._packet.frame_bgr.shape,
            }


class RotatingNDJSONSink:
    """Batch-buffered, size-rotating NDJSON telemetry and detection writer."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enable_rotation: bool = True,
        max_file_size_bytes: int = 100 * 1024 * 1024,
        batch_size: int = 10,
        prefix: str = "detections",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_rotation = bool(enable_rotation)
        self.max_file_size_bytes = max(1, int(max_file_size_bytes))
        self.batch_size = max(1, int(batch_size))
        self.prefix = prefix

        self._session_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file_index = 0
        self._buffer: list[str] = []
        self._bytes_written = 0
        self._current_path = self._build_path()
        self._fh = self._current_path.open("a", encoding="utf-8", buffering=8192)
        if self._current_path.exists():
            self._bytes_written = self._current_path.stat().st_size

    def _build_path(self) -> Path:
        if self.enable_rotation:
            return (
                self.output_dir
                / f"{self.prefix}_{self._session_tag}_{self._file_index:03d}.ndjson"
            )
        return self.output_dir / f"{self.prefix}.ndjson"

    @property
    def current_path(self) -> Path:
        return self._current_path

    @property
    def file_index(self) -> int:
        return self._file_index

    def write(self, record: InferenceRecord | Mapping[str, Any]) -> None:
        payload = record.to_dict() if isinstance(record, InferenceRecord) else record
        self._buffer.append(dumps_json_line(payload))
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or self._fh is None:
            return
        if self.enable_rotation and self._bytes_written >= self.max_file_size_bytes:
            self._rotate()
        chunk = "\n".join(self._buffer) + "\n"
        encoded_len = len(chunk.encode("utf-8"))
        self._fh.write(chunk)
        self._fh.flush()
        self._bytes_written += encoded_len
        self._buffer.clear()

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._file_index += 1
        self._current_path = self._build_path()
        self._fh = self._current_path.open("a", encoding="utf-8", buffering=8192)
        self._bytes_written = 0

    def close(self) -> None:
        if getattr(self, "_fh", None) is not None:
            try:
                self.flush()
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> RotatingNDJSONSink:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


class MultiStreamPipeline:
    """
    Coordinates N-stream zero-buffer RTSP decoding, single-slot `FrameSlot` sampling,
    hybrid CPU/Vulkan NCNN inference, optional detection-gated sign-crop CRNN OCR, and NDJSON output.
    """

    def __init__(
        self,
        config: PipelineConfig,
        router: HybridModelRouter,
        *,
        ocr_processor: CropOCRProcessor | None = None,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.config = config
        self.router = router
        self.ocr_processor = ocr_processor
        self.capture_factory = capture_factory or default_opencv_capture_factory

        self.num_streams = len(config.rtsp_urls)
        self.slots = tuple(FrameSlot(stream_id=i) for i in range(self.num_streams))
        self.result_queue: queue.Queue[InferenceRecord] = queue.Queue(
            maxsize=config.result_queue_maxsize
        )
        self._postprocess_queue: queue.Queue[tuple[Any, ...]] = queue.Queue(
            maxsize=config.result_queue_maxsize
        )
        self.stop_event = threading.Event()
        self._cpu_wake_event = threading.Event()
        self._gpu_wake_event = threading.Event()

        self._cpu_slot_ids: tuple[int, ...] = tuple(
            i for i in range(self.num_streams) if self.router.backend_for_stream(i) != "GPU"
        )
        self._gpu_slot_ids: tuple[int, ...] = tuple(
            i for i in range(self.num_streams) if self.router.backend_for_stream(i) == "GPU"
        )
        for i in self._cpu_slot_ids:
            self.slots[i].set_notify_event(self._cpu_wake_event)
        for i in self._gpu_slot_ids:
            self.slots[i].set_notify_event(self._gpu_wake_event)

        self._stats_lock = threading.Lock()
        self._decode_counts = [0] * self.num_streams
        self._sample_counts = [0] * self.num_streams
        self._inference_counts = [0] * self.num_streams
        self._detection_counts = [0] * self.num_streams
        self._reconnect_counts = [0] * self.num_streams
        self._connected = [False] * self.num_streams
        self._queue_drops = 0
        self._postprocess_drops = 0
        self._recent_latencies_ms: deque[float] = deque(maxlen=100)
        self._recent_inference_ms: deque[float] = deque(maxlen=100)

    def snapshot_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "num_streams": self.num_streams,
                "total_inferences": sum(self._inference_counts),
                "total_detections": sum(self._detection_counts),
                "queue_drops": self._queue_drops,
                "postprocess_drops": self._postprocess_drops,
                "overwrite_drops": [slot.overwrite_drops for slot in self.slots],
                "avg_latency_ms": (
                    float(np.mean(self._recent_latencies_ms))
                    if self._recent_latencies_ms
                    else 0.0
                ),
                "avg_inference_ms": (
                    float(np.mean(self._recent_inference_ms))
                    if self._recent_inference_ms
                    else 0.0
                ),
                "streams": [
                    {
                        "stream_id": i,
                        "backend": self.router.backend_for_stream(i),
                        "connected": self._connected[i],
                        "decoded_frames": self._decode_counts[i],
                        "sampled_frames": self._sample_counts[i],
                        "inferred_frames": self._inference_counts[i],
                        "detections": self._detection_counts[i],
                        "reconnects": self._reconnect_counts[i],
                        "overwrite_drops": self.slots[i].overwrite_drops,
                    }
                    for i in range(self.num_streams)
                ],
            }

    def _finalize_record(
        self,
        packet: FramePacket,
        detections: list[Detection],
        pre_ms: float,
        fwd_ms: float,
        post_ms: float,
        det_elapsed_ms: float,
        backend: str,
        *,
        run_ocr: bool = True,
        save_image: bool = True,
    ) -> InferenceRecord:
        ocr_ms = 0.0
        if run_ocr and self.ocr_processor is not None and detections:
            t_ocr = time.perf_counter()
            try:
                detections = self.ocr_processor.annotate(packet.frame_bgr, detections)
            except Exception:
                logger.exception("OCR postprocessing failed for stream %d", packet.stream_id)
            ocr_ms = (time.perf_counter() - t_ocr) * 1000.0

        t_end_mono = time.perf_counter()
        t_end_wall = time.time()
        # Exclude async queue dwell time so inference_time_ms reflects compute latency only.
        total_ms = det_elapsed_ms + ocr_ms
        if packet.capture_monotonic is not None:
            latency_ms = max(0.0, (t_end_mono - packet.capture_monotonic) * 1000.0)
        elif packet.capture_timestamp > 1_000_000_000.0:
            latency_ms = max(0.0, (t_end_wall - packet.capture_timestamp) * 1000.0)
        else:
            latency_ms = total_ms

        record = InferenceRecord(
            stream_id=packet.stream_id,
            frame_count=packet.frame_count,
            capture_timestamp=packet.capture_timestamp,
            inference_timestamp=t_end_wall,
            latency_ms=latency_ms,
            preprocess_ms=pre_ms,
            forward_ms=fwd_ms,
            postprocess_ms=post_ms,
            ocr_ms=ocr_ms,
            inference_time_ms=total_ms,
            backend=backend,
            detections=tuple(detections),
        )

        with self._stats_lock:
            self._inference_counts[packet.stream_id] += 1
            self._detection_counts[packet.stream_id] += len(detections)
            self._recent_latencies_ms.append(latency_ms)
            self._recent_inference_ms.append(total_ms)

        if save_image and self.config.save_images and detections:
            try:
                self._save_annotated_frame(packet, detections)
            except Exception:
                logger.exception("Annotated frame save failed for stream %d", packet.stream_id)

        try:
            self.result_queue.put_nowait(record)
        except queue.Full:
            with self._stats_lock:
                self._queue_drops += 1

        return record

    def _process_slot(
        self,
        slot: FrameSlot,
        *,
        async_postprocess: bool = False,
    ) -> tuple[bool, InferenceRecord | None]:
        packet = slot.take_if_new()
        if packet is None:
            return False, None

        t_start_mono = time.perf_counter()
        detector = self.router.select_detector(packet.stream_id)
        backend = self.router.backend_for_stream(packet.stream_id)
        detections, pre_ms, fwd_ms, post_ms = detector.detect_timed(
            packet.frame_bgr
        )
        det_elapsed_ms = (time.perf_counter() - t_start_mono) * 1000.0

        needs_postprocess = bool(
            detections and (self.ocr_processor is not None or self.config.save_images)
        )
        if async_postprocess and needs_postprocess:
            try:
                self._postprocess_queue.put_nowait(
                    (packet, detections, pre_ms, fwd_ms, post_ms, det_elapsed_ms, backend)
                )
                return True, None
            except queue.Full:
                with self._stats_lock:
                    self._postprocess_drops += 1
                rec = self._finalize_record(
                    packet,
                    detections,
                    pre_ms,
                    fwd_ms,
                    post_ms,
                    det_elapsed_ms,
                    backend,
                    run_ocr=False,
                    save_image=False,
                )
                return True, rec

        rec = self._finalize_record(
            packet,
            detections,
            pre_ms,
            fwd_ms,
            post_ms,
            det_elapsed_ms,
            backend,
            run_ocr=True,
            save_image=True,
        )
        return True, rec

    def process_available_slots_once(self) -> list[InferenceRecord]:
        """
        Poll each stream's `FrameSlot` once and run inference + postprocessing synchronously.
        Useful for deterministic unit tests and single-step evaluation.
        """
        records: list[InferenceRecord] = []
        for slot in self.slots:
            _, rec = self._process_slot(slot, async_postprocess=False)
            if rec is not None:
                records.append(rec)
        return records

    def _save_annotated_frame(
        self,
        packet: FramePacket,
        detections: Sequence[Detection],
    ) -> Path | None:
        """Render bounding boxes onto a copy of `packet.frame_bgr` and save to `<output_dir>/images/`."""
        image_dir = Path(self.config.output_dir) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        canvas = packet.frame_bgr.copy()
        h, w = canvas.shape[:2]
        for det in detections:
            x1 = max(0, min(round(det.bbox[0]), w - 1))
            y1 = max(0, min(round(det.bbox[1]), h - 1))
            x2 = max(0, min(round(det.bbox[2]), w - 1))
            y2 = max(0, min(round(det.bbox[3]), h - 1))
            if x2 > x1 and y2 > y1:
                canvas[y1 : min(y1 + 2, h), x1:x2] = (0, 255, 0)
                canvas[max(0, y2 - 2) : y2, x1:x2] = (0, 255, 0)
                canvas[y1:y2, x1 : min(x1 + 2, w)] = (0, 255, 0)
                canvas[y1:y2, max(0, x2 - 2) : x2] = (0, 255, 0)

        out_path = (
            image_dir
            / f"stream{packet.stream_id}_frame{packet.frame_count:06d}.npy"
        )
        try:
            import cv2
            jpg_path = out_path.with_suffix(".jpg")
            cv2.imwrite(str(jpg_path), canvas)
            return jpg_path
        except ImportError:
            np.save(out_path, canvas)
            return out_path

    def _decoder_loop(self, stream_id: int, url: str) -> None:
        target_interval = 1.0 / self.config.target_fps_per_stream
        last_sample_time = 0.0
        sample_counter = 0

        try:
            import cv2
            prop_buffersize = cv2.CAP_PROP_BUFFERSIZE
        except ImportError:
            prop_buffersize = 38

        while not self.stop_event.is_set():
            cap = self.capture_factory(url)
            if not cap.isOpened():
                with self._stats_lock:
                    self._reconnect_counts[stream_id] += 1
                    self._connected[stream_id] = False
                cap.release()
                self.stop_event.wait(self.config.reconnect_delay_sec)
                continue

            try:
                cap.set(prop_buffersize, float(self.config.cap_buffersize))
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Stream %d: unable to set CAP_PROP_BUFFERSIZE: %s", stream_id, exc)

            with self._stats_lock:
                self._connected[stream_id] = True

            try:
                while not self.stop_event.is_set():
                    try:
                        ok, frame = cap.read()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Stream %d read error: %s", stream_id, exc)
                        ok, frame = False, None

                    if not ok or frame is None:
                        with self._stats_lock:
                            self._connected[stream_id] = False
                            self._reconnect_counts[stream_id] += 1
                        break

                    now_mono = time.perf_counter()
                    with self._stats_lock:
                        self._decode_counts[stream_id] += 1

                    if (now_mono - last_sample_time) >= target_interval:
                        sample_counter += 1
                        last_sample_time = now_mono
                        self.slots[stream_id].put(
                            frame_bgr=frame,
                            capture_timestamp=time.time(),
                            frame_count=sample_counter,
                            stream_id=stream_id,
                            capture_monotonic=now_mono,
                        )
                        with self._stats_lock:
                            self._sample_counts[stream_id] = sample_counter
            finally:
                cap.release()
            if not self.stop_event.is_set():
                self.stop_event.wait(self.config.reconnect_delay_sec)

    def _backend_inference_loop(
        self,
        slot_ids: Sequence[int],
        wake_event: threading.Event,
        *,
        async_postprocess: bool,
    ) -> None:
        while not self.stop_event.is_set():
            wake_event.clear()
            processed_any = False
            for stream_id in slot_ids:
                if self.stop_event.is_set():
                    break
                try:
                    taken, _ = self._process_slot(
                        self.slots[stream_id],
                        async_postprocess=async_postprocess,
                    )
                except Exception:
                    logger.exception("Inference worker error on stream %d", stream_id)
                    continue
                if taken:
                    processed_any = True
            if not processed_any and not self.stop_event.is_set():
                wake_event.wait(timeout=0.05)

    def _postprocess_loop(self, postprocess_stop_event: threading.Event) -> None:
        while not (postprocess_stop_event.is_set() and self._postprocess_queue.empty()):
            try:
                (
                    packet,
                    detections,
                    pre_ms,
                    fwd_ms,
                    post_ms,
                    det_elapsed_ms,
                    backend,
                ) = self._postprocess_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._finalize_record(
                    packet,
                    detections,
                    pre_ms,
                    fwd_ms,
                    post_ms,
                    det_elapsed_ms,
                    backend,
                    run_ocr=True,
                    save_image=True,
                )
            except Exception:
                logger.exception("Postprocess worker error on stream %d", packet.stream_id)

    def _writer_loop(
        self,
        sink: RotatingNDJSONSink,
        writer_stop_event: threading.Event | None = None,
    ) -> None:
        stop_ev = writer_stop_event if writer_stop_event is not None else self.stop_event
        last_flush = time.perf_counter()
        while not (stop_ev.is_set() and self.result_queue.empty()):
            try:
                record = self.result_queue.get(timeout=0.1)
                sink.write(record)
            except queue.Empty:
                pass
            now = time.perf_counter()
            if (now - last_flush) >= self.config.json_flush_interval_sec:
                sink.flush()
                last_flush = now

        sink.flush()

    def run(self, max_duration_sec: float | None = None) -> dict[str, Any]:
        """Run the full multi-stream pipeline until interrupted or `max_duration_sec` elapses."""
        self.stop_event.clear()
        self._cpu_wake_event.clear()
        self._gpu_wake_event.clear()
        postprocess_stop_event = threading.Event()
        writer_stop_event = threading.Event()
        configure_ffmpeg_capture_env(self.config.opencv_ffmpeg_options, overwrite=True)

        prev_sigterm: Any = None
        in_main_thread = threading.current_thread() is threading.main_thread()
        if in_main_thread:
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, lambda _sig, _frm: self.stop_event.set())

        use_async_postprocess = (
            self.ocr_processor is not None or self.config.save_images
        )

        max_bytes = int(self.config.max_json_file_size_mb * 1024 * 1024)
        with RotatingNDJSONSink(
            self.config.output_dir,
            enable_rotation=self.config.enable_log_rotation,
            max_file_size_bytes=max_bytes,
            batch_size=self.config.json_batch_size,
        ) as sink:
            decoder_threads = [
                threading.Thread(
                    target=self._decoder_loop,
                    args=(i, url),
                    daemon=True,
                    name=f"RTSPDecoder-{i}",
                )
                for i, url in enumerate(self.config.rtsp_urls)
            ]
            inference_threads: list[threading.Thread] = []
            if self._cpu_slot_ids:
                inference_threads.append(
                    threading.Thread(
                        target=self._backend_inference_loop,
                        args=(self._cpu_slot_ids, self._cpu_wake_event),
                        kwargs={"async_postprocess": use_async_postprocess},
                        daemon=True,
                        name="NCNNWorker-CPU",
                    )
                )
            if self._gpu_slot_ids:
                inference_threads.append(
                    threading.Thread(
                        target=self._backend_inference_loop,
                        args=(self._gpu_slot_ids, self._gpu_wake_event),
                        kwargs={"async_postprocess": use_async_postprocess},
                        daemon=True,
                        name="NCNNWorker-GPU",
                    )
                )
            postprocess_thread = (
                threading.Thread(
                    target=self._postprocess_loop,
                    args=(postprocess_stop_event,),
                    daemon=True,
                    name="PostprocessWorker",
                )
                if use_async_postprocess
                else None
            )
            writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(sink, writer_stop_event),
                daemon=True,
                name="NDJSONWriter",
            )

            for thread in decoder_threads:
                thread.start()
            for thread in inference_threads:
                thread.start()
            if postprocess_thread is not None:
                postprocess_thread.start()
            writer_thread.start()

            try:
                self.stop_event.wait(timeout=max_duration_sec)
            except KeyboardInterrupt:
                self.stop_event.set()
            finally:
                self.stop_event.set()
                self._cpu_wake_event.set()
                self._gpu_wake_event.set()
                if in_main_thread and prev_sigterm is not None:
                    signal.signal(signal.SIGTERM, prev_sigterm)
                decoder_deadline = time.perf_counter() + 6.0
                for thread in decoder_threads:
                    thread.join(timeout=max(0.0, decoder_deadline - time.perf_counter()))
                infer_deadline = time.perf_counter() + 5.0
                for thread in inference_threads:
                    thread.join(timeout=max(0.0, infer_deadline - time.perf_counter()))
                if postprocess_thread is not None:
                    postprocess_stop_event.set()
                    postprocess_thread.join(timeout=5.0)
                writer_stop_event.set()
                writer_thread.join(timeout=10.0)

        return self.snapshot_stats()
