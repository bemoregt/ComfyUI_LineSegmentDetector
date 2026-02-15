# ComfyUI Line Segment Detector

A ComfyUI custom node pack that detects line segments in images using the **OpenCV LSD (Line Segment Detector)** algorithm and provides utilities for visualization, JSON export, and statistics.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_LineSegmentDetector/blob/main/ScrShot%2010.png)

## Nodes

### 1. Line Segment Detector (LSD)

Detects line segments from an input image using the LSD algorithm.

**Inputs**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Input image (batch supported) |
| `scale` | FLOAT | 0.8 | Internal image scale for LSD. Lower values are faster. |
| `sigma_scale` | FLOAT | 0.6 | Gaussian sigma = `sigma_scale / scale` |
| `quant` | FLOAT | 2.0 | Bound on gradient norm quantization error |
| `ang_th` | FLOAT | 22.5 | Gradient angle tolerance in degrees |
| `log_eps` | FLOAT | 0.0 | Detection threshold: −log₁₀(NFA) > log_eps |
| `density_th` | FLOAT | 0.7 | Minimum density of aligned points in the enclosing rectangle |
| `min_length` | FLOAT | 10.0 | Discard segments shorter than this value (pixels) |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `line_segments` | LINE_SEGMENT | List of detected segments (see data format below) |

---

### 2. Line Segment Visualizer

Draws detected line segments on top of the original image.

**Inputs**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Background image to draw on |
| `line_segments` | LINE_SEGMENT | — | Segments to draw |
| `color_r` | INT | 0 | Line color — Red channel (0–255) |
| `color_g` | INT | 255 | Line color — Green channel (0–255) |
| `color_b` | INT | 0 | Line color — Blue channel (0–255) |
| `thickness` | INT | 1 | Line thickness in pixels |

**Outputs**: `IMAGE`

---

### 3. Line Segment → JSON

Serializes the `LINE_SEGMENT` data to a JSON string.

**Inputs**: `line_segments` (LINE_SEGMENT), `indent` (INT, default 2)

**Outputs**: `STRING`

---

### 4. Line Segment Stats

Computes summary statistics from detected segments.

**Inputs**: `line_segments` (LINE_SEGMENT)

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `count` | INT | Number of detected segments |
| `mean_length` | FLOAT | Average segment length (px) |
| `max_length` | FLOAT | Longest segment length (px) |
| `min_length` | FLOAT | Shortest segment length (px) |
| `stats_json` | STRING | Full stats as a JSON string (includes std_length) |

---

## LINE_SEGMENT Data Format

The `LINE_SEGMENT` type is a plain Python `list` of `dict` objects — fully JSON-serializable and easy to process downstream.

```json
[
  {
    "x1": 10.3,
    "y1": 12.7,
    "x2": 118.5,
    "y2": 120.1,
    "length": 152.4,
    "angle": 44.9,
    "width": 1.8,
    "prec": 0.78,
    "nfa": -3.2
  },
  ...
]
```

| Field | Description |
|-------|-------------|
| `x1`, `y1` | Start point (pixels) |
| `x2`, `y2` | End point (pixels) |
| `length` | Euclidean length in pixels |
| `angle` | Angle in degrees (−180° to 180°, measured from positive x-axis) |
| `width` | Estimated line width |
| `prec` | Alignment precision |
| `nfa` | −log₁₀(NFA) detection score |

For batch inputs (more than one image), the output is a `list` of the above lists — one per image in the batch.

---

## Requirements

- `opencv-python >= 4.0`
- `numpy`
- `torch`

No extra model weights are needed — LSD is a classical algorithm built into OpenCV.

```bash
pip install opencv-python
```

---

## Installation

Copy or symlink this folder into your ComfyUI `custom_nodes` directory:

```bash
# Option A — copy
cp -r ComfyUI_LineSegmentDetector /path/to/ComfyUI/custom_nodes/

# Option B — symlink
ln -s /absolute/path/to/ComfyUI_LineSegmentDetector \
      /path/to/ComfyUI/custom_nodes/ComfyUI_LineSegmentDetector
```

Restart ComfyUI. The nodes will appear under the **`image/analysis`** category.

---

## Example Workflow

```
Load Image
    │
    ▼
Line Segment Detector (LSD)   ←── tune scale / min_length
    │
    ├──► Line Segment Visualizer ──► Preview Image
    │
    ├──► Line Segment Stats      ──► count / mean_length / …
    │
    └──► Line Segment → JSON     ──► downstream text node
```

---

## License

MIT
