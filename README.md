# FencerAI - The Spatio-Temporal Fencing AI Coach
*Version: 1.2 | Last Updated: 2026-03-28*

FencerAI is an **edge-first, real-time computer vision system** designed to analyze fencing bouts using a single RGB camera. It extracts a **101-dimensional spatio-temporal feature vector** to evaluate distances, tempos, and actions.

## Vision

FencerAI is not a generic pose estimation app. It is a **high-performance spatio-temporal engine** for competitive fencing analysis that:

- Captures explosive fencing movements with **<500ms latency**
- Tracks **dual fencers** while filtering out referees
- Extracts **canonicalized features** (left-fencer perspective) for downstream classification
- Provides **real-time health monitoring** and visualization tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FencerAI Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Video/Audio Input                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                       │
│  │ TimestampedBuffer │ ◄── Audio-video sync via cross-correlation          │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │    RTMPose      │ ◄── 17-keypoint COCO format (rtmlib ONNX runtime)    │
│  │  (Lightweight)   │      Modes: lightweight | balanced | performance     │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │  Norfair Dual    │ ◄── Pose embedding similarity matching                │
│  │  Tracker         │      Referee filter: bottom 70% Y-axis               │
│  │  + EMA Predictor │      Graceful failure handling                        │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │   Calibrator    │ ◄── Homography (pixel → meter transformation)        │
│  │  (Optional)      │                                                       │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │   Audio Event   │ ◄── Energy-based blade touch detection                │
│  │   Detection     │                                                       │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │   Feature       │ ◄── 101-dim vector per fencer (canonicalized)         │
│  │   Extractor     │      EMA-smoothed velocity & acceleration             │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │  (N, 2, 101)    │ ◄── Feature Matrix: N frames × 2 fencers × 101 dims  │
│  │  Numpy Matrix   │                                                       │
│  └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Pipeline
- **RTMPose Integration**: Lightweight pose estimation (158ms/frame on CPU)
- **Dual Fencer Tracking**: Norfair-based tracker with pose embedding similarity
- **Referee Filtering**: Automatic filtering of detections in upper 30% of frame
- **Graceful Failure**: EMA-predicted positions when tracking is lost
- **Audio Events**: Energy-based blade touch detection

### Feature Extraction
- **101-Dimensional Vector**: Complete feature dictionary per frame
- **Canonicalization**: Left-fencer perspective (no left/right ambiguity)
- **Temporal Derivatives**: EMA-smoothed velocity (24 dims) and acceleration (24 dims)
- **Physical Metrics**: Homography-based meter conversion for distance features

### Visualization & Monitoring
- **`--visualize`**: Skeleton overlay video with fencer color coding
- **`--heatmap`**: Feature matrix heatmap export (per-fencer + combined)
- **Health Monitor**: Real-time detection quality, confidence, and latency tracking

### Data Integrity
- **Pydantic Validation**: All inter-layer data validated via schemas
- **Type Safety**: Python 3.9+ type hints throughout
- **Temporal Normalization**: Actual timestamps for derivative calculations

## 101-Dimensional Feature Dictionary

Each frame produces a **101-dimensional vector for each fencer**:

| Index | Category | Dimensions | Description |
|-------|----------|-------------|-------------|
| 0-23 | Static Geometry | 24 | 12 keypoints (x,y), shoulder-width normalized |
| 24-25 | Center of Mass | 2 | Pelvis/hip center (x, y) |
| 26-36 | Distance | 11 | Inter-fencer distance, stance width (meters) |
| 37-40 | Angular | 4 | Knee angles, weapon elbow, torso lean |
| 41-42 | Torso Orientation | 2 | Shoulder-to-hip vector direction |
| 43-48 | Arm Extension | 6 | Weapon arm extension ratio + angle |
| 49-72 | Velocity | 24 | 1st derivative of static geometry (EMA smoothed) |
| 73-96 | Acceleration | 24 | 2nd derivative of static geometry |
| 97-98 | CoM Velocity | 2 | Center of mass velocity |
| 99 | CoM Acceleration | 1 | Center of mass acceleration |
| 100 | Audio Flag | 1 | Blade touch detection (1.0=touch, 0.0=silence) |

## Directory Structure

```
fencer_ai/
├── src/
│   ├── perception/           # Perception Layer
│   │   ├── rtmpose.py        # RTMPose wrapper (rtmlib)
│   │   ├── tracker.py       # Norfair dual-tracker + referee filter
│   │   ├── calibrator.py    # Homography calibration
│   │   ├── audio.py         # Audio event detection
│   │   ├── audio_buffer.py  # Circular audio buffer
│   │   └── pipeline.py      # Perception orchestration
│   ├── recognition/          # Recognition Layer
│   │   ├── feature_math.py   # Vectorized numpy geometry
│   │   └── feature_extractor.py  # 101-dim extraction + canonicalization
│   ├── utils/                # Utilities
│   │   ├── schemas.py        # Pydantic models (FencerPose, FrameData, etc.)
│   │   ├── constants.py      # Indices, COCO_SKELETON_CONNECTIONS
│   │   ├── buffer.py         # TimestampedBuffer
│   │   ├── profiling.py      # LatencyProfiler, MemoryProfiler, HealthMonitor
│   │   ├── visualization.py  # Skeleton drawing, heatmap export
│   │   ├── config.py         # Configuration management
│   │   ├── logging.py        # Loguru with temporal annotations
│   │   └── types.py          # Type aliases
│   └── main_pipeline.py      # CLI entry point
├── tests/                    # Test suite
│   ├── unit/                # Unit tests (288 tests)
│   ├── test_integration_pipeline.py
│   └── test_e2e_clean_bout.py  # E2E tests with sample video
├── docs/
│   ├── ARCHITECTURE.md       # Detailed architecture
│   ├── DEVELOPMENT_PROGRESS.md  # Phase tracking
│   └── ARCHITECTURAL_DECISIONS.md  # Design decisions (11 ADs)
├── data/samples/             # Sample videos
└── requirements.txt         # Dependencies
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run feature extraction
python -m src.main_pipeline \
    --video data/samples/498_1728950803.mp4 \
    --output outputs/features

# 3. With visualization
python -m src.main_pipeline \
    --video data/samples/498_1728950803.mp4 \
    --output outputs/features \
    --visualize

# 4. With heatmap export
python -m src.main_pipeline \
    --video data/samples/498_1728950803.mp4 \
    --output outputs/features \
    --heatmap

# 5. With profiling
python -m src.main_pipeline \
    --video data/samples/498_1728950803.mp4 \
    --output outputs/features \
    --profile
```

## Output

The pipeline produces:

```
outputs/
├── features.npy              # (N_frames, 2, 101) feature matrix
├── features.json             # Metadata (timestamps, frame_ids, etc.)
├── features_visualization.mp4 # Skeleton overlay (with --visualize)
├── features_fencer0_heatmap.png  # Fencer 0 heatmap (with --heatmap)
├── features_fencer1_heatmap.png  # Fencer 1 heatmap
└── features_combined_heatmap.png # Combined heatmap
```

## Performance

| RTMPose Mode | Latency (Mean) | Latency (Min) | Latency (Max) |
|--------------|----------------|---------------|---------------|
| **Lightweight** | **158.8ms** | 151.6ms | 172.1ms |
| Balanced | 1214.4ms | 1184.8ms | 1381.4ms |
| Performance | 4797.6ms | 4695.9ms | 4940.9ms |

**Lightweight mode is the default** (8x faster than balanced).

## Test Coverage

- **288 unit tests** covering all modules
- **6 E2E tests** with sample video validation
- Tests for schemas, buffer, tracker, feature math, canonicalization, EMA

## Development Progress

See [DEVELOPMENT_PROGRESS.md](docs/DEVELOPMENT_PROGRESS.md) for detailed task tracking.

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Foundation | 11 | ✅ Complete |
| Phase 2: Perception | 14 | ✅ Complete |
| Phase 3: Recognition | 18 | ✅ Complete |
| Phase 4: Pipeline | 9 | ✅ Complete |
| Phase 5: Testing | 10 | 8/10 (E2E pending videos) |
| Phase 6: Optimization | 5 | ✅ Complete |
| Phase 7: Visualization | 2 | ✅ Complete |

## Dependencies

- **rtmlib**: RTMPose ONNX runtime
- **onnxruntime**: ONNX model inference
- **norfair**: Multi-person tracking
- **pydantic**: Data validation
- **numpy**: Vectorized operations
- **opencv-python**: Video I/O, visualization
- **scipy**: Signal processing
- **librosa**: Audio analysis
- **psutil**: System monitoring
