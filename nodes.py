import cv2
import numpy as np
import torch
import json


# ──────────────────────────────────────────────
# LINE_SEGMENT 타입: List[dict] 형태
#   각 dict = {"x1": float, "y1": float, "x2": float, "y2": float,
#              "length": float, "angle": float}
# ──────────────────────────────────────────────


class LineSegmentDetectorNode:
    """OpenCV LSD(Line Segment Detector)를 사용해 이미지에서 선분을 검출합니다.

    출력 LINE_SEGMENT 는 검출된 선분 리스트(JSON 직렬화 가능한 dict 리스트)입니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # [B, H, W, C] float32 0~1
                "scale":  ("FLOAT",  {"default": 0.8,  "min": 0.1, "max": 2.0,  "step": 0.05,
                                       "tooltip": "LSD 내부 이미지 스케일 (작을수록 빠름)"}),
                "sigma_scale": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.05,
                                           "tooltip": "가우시안 시그마 = sigma_scale / scale"}),
                "quant":  ("FLOAT",  {"default": 2.0,  "min": 0.0, "max": 10.0, "step": 0.5,
                                       "tooltip": "양자화 오류 허용치"}),
                "ang_th": ("FLOAT",  {"default": 22.5, "min": 1.0, "max": 90.0, "step": 0.5,
                                       "tooltip": "각도 허용 오차 (도)"}),
                "log_eps": ("FLOAT", {"default": 0.0,  "min": -5.0, "max": 5.0, "step": 0.5,
                                       "tooltip": "검출 임계값 (log NFA)"}),
                "density_th": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "정렬 포인트 밀도 임계값"}),
                "min_length": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 9999.0, "step": 1.0,
                                          "tooltip": "이 픽셀 길이 미만 선분 제거"}),
            },
        }

    RETURN_TYPES = ("LINE_SEGMENT",)
    RETURN_NAMES = ("line_segments",)
    FUNCTION = "detect"
    CATEGORY = "image/analysis"

    def detect(self, image, scale, sigma_scale, quant, ang_th,
               log_eps, density_th, min_length):

        lsd = cv2.createLineSegmentDetector(
            refine=cv2.LSD_REFINE_STD,
            scale=scale,
            sigma_scale=sigma_scale,
            quant=quant,
            ang_th=ang_th,
            log_eps=log_eps,
            density_th=density_th,
            n_bins=1024,
        )

        batch_results = []
        batch_size = image.shape[0]

        for i in range(batch_size):
            # ComfyUI IMAGE [H, W, C] float32 0~1 → grayscale uint8
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            lines, widths, precs, nfa = lsd.detect(gray)

            segments = []
            if lines is not None:
                for idx, line in enumerate(lines):
                    x1, y1, x2, y2 = float(line[0][0]), float(line[0][1]), \
                                      float(line[0][2]), float(line[0][3])
                    length = float(np.hypot(x2 - x1, y2 - y1))
                    if length < min_length:
                        continue
                    angle  = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    width  = float(widths[idx][0]) if widths is not None else 1.0
                    prec   = float(precs[idx][0])  if precs  is not None else 0.0
                    nfa_v  = float(nfa[idx][0])    if nfa    is not None else 0.0
                    segments.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "length": length,
                        "angle":  angle,
                        "width":  width,
                        "prec":   prec,
                        "nfa":    nfa_v,
                    })

            batch_results.append(segments)

        # 배치 1장이면 단일 리스트, 여러 장이면 리스트의 리스트
        result = batch_results[0] if batch_size == 1 else batch_results
        return (result,)


# ──────────────────────────────────────────────
# LineSegment → 시각화 이미지
# ──────────────────────────────────────────────

class LineSegmentVisualizer:
    """LINE_SEGMENT 데이터를 원본 이미지 위에 그려서 반환합니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":         ("IMAGE",),
                "line_segments": ("LINE_SEGMENT",),
                "color_r": ("INT", {"default": 0,   "min": 0, "max": 255, "step": 1}),
                "color_g": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "color_b": ("INT", {"default": 0,   "min": 0, "max": 255, "step": 1}),
                "thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized_image",)
    FUNCTION = "visualize"
    CATEGORY = "image/analysis"

    def visualize(self, image, line_segments, color_r, color_g, color_b, thickness):
        # 단일 배치 처리
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8).copy()
        canvas = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 배치 결과인 경우 첫 번째 프레임 사용
        segs = line_segments[0] if (line_segments and isinstance(line_segments[0], list)) \
               else line_segments

        color = (color_b, color_g, color_r)  # OpenCV BGR
        for seg in segs:
            pt1 = (int(round(seg["x1"])), int(round(seg["y1"])))
            pt2 = (int(round(seg["x2"])), int(round(seg["y2"])))
            cv2.line(canvas, pt1, pt2, color, thickness)

        result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (tensor,)


# ──────────────────────────────────────────────
# LineSegment → JSON 문자열
# ──────────────────────────────────────────────

class LineSegmentToJSON:
    """LINE_SEGMENT 데이터를 JSON 문자열로 변환합니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line_segments": ("LINE_SEGMENT",),
                "indent": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_string",)
    FUNCTION = "to_json"
    CATEGORY = "image/analysis"

    def to_json(self, line_segments, indent):
        text = json.dumps(line_segments, ensure_ascii=False, indent=indent if indent > 0 else None)
        return (text,)


# ──────────────────────────────────────────────
# LineSegment → 통계 정보
# ──────────────────────────────────────────────

class LineSegmentStats:
    """LINE_SEGMENT 에서 통계(개수, 평균 길이, 최대/최소 길이 등)를 추출합니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line_segments": ("LINE_SEGMENT",),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("count", "mean_length", "max_length", "min_length", "stats_json")
    FUNCTION = "compute"
    CATEGORY = "image/analysis"

    def compute(self, line_segments):
        segs = line_segments[0] if (line_segments and isinstance(line_segments[0], list)) \
               else line_segments

        if not segs:
            return (0, 0.0, 0.0, 0.0, "{}")

        lengths = [s["length"] for s in segs]
        stats = {
            "count":       len(segs),
            "mean_length": float(np.mean(lengths)),
            "max_length":  float(np.max(lengths)),
            "min_length":  float(np.min(lengths)),
            "std_length":  float(np.std(lengths)),
        }
        return (
            stats["count"],
            stats["mean_length"],
            stats["max_length"],
            stats["min_length"],
            json.dumps(stats, ensure_ascii=False, indent=2),
        )


# ──────────────────────────────────────────────
# Node registration
# ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LineSegmentDetector":   LineSegmentDetectorNode,
    "LineSegmentVisualizer": LineSegmentVisualizer,
    "LineSegmentToJSON":     LineSegmentToJSON,
    "LineSegmentStats":      LineSegmentStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LineSegmentDetector":   "Line Segment Detector (LSD)",
    "LineSegmentVisualizer": "Line Segment Visualizer",
    "LineSegmentToJSON":     "Line Segment → JSON",
    "LineSegmentStats":      "Line Segment Stats",
}
