# ProctorVision

Real-time computer vision system for detecting suspicious exam behavior using low-latency inference and streaming event aggregation.

## Overview

ProctorVision is a CPU-optimized, real-time CV pipeline that performs:

* Face detection (MediaPipe)
* Gaze estimation (lightweight intensity-based method)
* Object detection (YOLOv8, ONNX optimized)

The system processes live frames, extracts signals, and aggregates them into session-level risk scores using a stateful event pipeline.

---

## System Architecture

```
Frontend (Stream / Dashboard)
        │
        ▼
FastAPI Backend (Async API)
        │
        ▼
Vision Pipeline (Per-frame inference)
        │
        ▼
Stateful Event Engine (Ray actors)
        │
        ▼
Session Risk Aggregation
```

### Key Design Decisions

* Stateless vision layer → fast signal generation
* Stateful worker layer → temporal smoothing and decision logic
* Asynchronous API → non-blocking frame ingestion
* CPU-first design → optimized for low-resource environments

---

## Core Components

### Vision Pipeline (`vision.py`)

* YOLOv8 (ONNX or PyTorch) for object detection
* MediaPipe for face detection
* Lightweight gaze estimation via pixel intensity
* Frame subsampling for reduced latency

### Event Engine (`workers.py`)

* Ray-based actor system for parallel processing
* Temporal smoothing using sliding windows (deques)
* Event cooldowns to prevent duplicate triggers
* Stateful session tracking and risk scoring

### API Layer (`server.py`)

* FastAPI-based async server
* Non-blocking frame ingestion (`/processFrame`)
* Polling-based result retrieval
* Session-level aggregation (`/sessionSummary`)

---

## Performance

| Stage            | Latency    |
| ---------------- | ---------- |
| Frame Capture    | ~10ms      |
| Vision Inference | 150–250ms  |
| Event Processing | ~5ms       |
| Total            | ~170–260ms |

**Throughput:** ~6–8 FPS (CPU)

### Optimizations

* ONNX inference for reduced model overhead
* Frame subsampling for expensive detections
* Removal of redundant smoothing layers
* Optional GPU/MPS acceleration
* Model fusion for PyTorch inference

---

## Detection Signals

| Signal        | Description                 |
| ------------- | --------------------------- |
| gazeAway      | Sustained head deviation    |
| faceMissing   | Face absent for time window |
| phoneDetected | Phone visible in frame      |
| multipleFaces | More than one face detected |

---

## Event Logic

Signals are converted into events using:

* Sliding window majority voting
* Cooldown timers per event type
* Weighted risk accumulation

This separates noisy per-frame predictions from stable behavioral events.

---

## Running the System

Start the API:

```bash
python3 server.py
```

Send a frame:

```json
POST /processFrame
{
  "frame": "<base64 image>"
}
```

Get session summary:

```bash
GET /sessionSummary
```

---

## Configuration

Environment variables:

```bash
PHONE_MODEL_PATH=yolov8n.onnx
PHONE_IMG_SIZE=416
PHONE_CHECK_INTERVAL=3
USE_GPU=auto
```

---

## Key Insights

* System is vision-bound, not I/O-bound
* Removing redundant smoothing reduces latency significantly
* Separating signal generation from decision logic improves stability
* ONNX + subsampling provides the best CPU performance tradeoff

---

## Future Improvements

* Batch inference for higher throughput
* GPU scheduling and queue-based execution
* Streaming pipeline (Kafka or gRPC)
* Landmark-based gaze estimation

---

## License

MIT
