# src/detector.py
# Multi-object detector: YOLOv8n for localization/person detection,
# Pokémon CNN for classification of cropped regions.
#
# Pipeline per frame:
#   1. YOLOv8n detects all objects + bounding boxes
#   2. "person" class → kept as-is (label "Person")
#   3. Other classes  → crop region → run through Pokémon CNN
#      • CNN conf > POKEMON_THRESHOLD  → label = Pokémon name
#      • CNN conf ≤ POKEMON_THRESHOLD  → label = YOLO class name
#   4. Returns List[Detection] for caller to render

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ── Lazy imports (avoid startup cost) ──────────────────────────────
_yolo_model  = None
_yolo_lock   = threading.Lock()

# ── Thresholds ─────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD    = 0.35   # minimum YOLO detection confidence
POKEMON_THRESHOLD      = 0.40   # CNN conf to override YOLO label with Pokémon name
PERSON_CLASS_ID        = 0      # COCO class 0 = "person"

# ── Color palette (BGR for OpenCV) ─────────────────────────────────
COLOR_PERSON  = (80, 220, 80)    # green
COLOR_POKEMON = (60,  60, 255)   # red
COLOR_OTHER   = (140, 140, 140)  # grey

# ── Color palette (Hex for CustomTkinter) ──────────────────────────
HEX_PERSON  = "#22C55E"
HEX_POKEMON = "#FF3E3E"
HEX_OTHER   = "#64748B"


@dataclass
class Detection:
    """Single detected object in a frame."""
    label:      str           # e.g. "Pikachu", "Person", "bottle"
    confidence: float         # 0.0 – 1.0
    box:        Tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame coords
    color_bgr:  Tuple[int, int, int] = field(default_factory=lambda: COLOR_OTHER)
    color_hex:  str = HEX_OTHER
    is_person:  bool = False
    is_pokemon: bool = False


# ── Singleton loader ────────────────────────────────────────────────

def _load_yolo():
    """Download + load YOLOv8n once (thread-safe)."""
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            # yolov8n.pt is auto-downloaded on first call (~6 MB)
            _yolo_model = YOLO("yolov8n.pt")
            _yolo_model.fuse()   # merge Conv+BN for faster inference


# ── Main detector ───────────────────────────────────────────────────

class ObjectDetector:
    """
    Stateless detector — create once, call detect() per frame.
    Thread-safe (YOLO and CNN models are read-only after loading).
    """

    def __init__(self):
        _load_yolo()
        # Ensure Pokémon CNN is loaded via predict_dl singleton
        import src.predict_dl as predict_dl
        predict_dl._load_model()
        self._predict_dl = predict_dl

    # ----------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR frame.

        Returns a (possibly empty) list of Detection objects,
        one per found object, sorted by confidence descending.
        """
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        results = _yolo_model(
            frame_bgr,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):
            xyxy       = boxes_data.xyxy[i].cpu().numpy().astype(int)
            yolo_conf  = float(boxes_data.conf[i].cpu())
            cls_id     = int(boxes_data.cls[i].cpu())
            cls_name   = _yolo_model.names[cls_id]

            x1, y1, x2, y2 = _clamp_box(xyxy, w, h)

            # ── Person ────────────────────────────────────────────
            if cls_id == PERSON_CLASS_ID:
                detections.append(Detection(
                    label="Person",
                    confidence=yolo_conf,
                    box=(x1, y1, x2, y2),
                    color_bgr=COLOR_PERSON,
                    color_hex=HEX_PERSON,
                    is_person=True,
                ))
                continue

            # ── Non-person: crop and run Pokémon CNN ──────────────
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            poke_name, poke_conf = self._classify_crop(crop)

            if poke_conf >= POKEMON_THRESHOLD:
                label     = f"{poke_name.capitalize()}"
                conf_show = poke_conf
                color_bgr = COLOR_POKEMON
                color_hex = HEX_POKEMON
                is_poke   = True
            else:
                label     = cls_name          # YOLO's own class name
                conf_show = yolo_conf
                color_bgr = COLOR_OTHER
                color_hex = HEX_OTHER
                is_poke   = False

            detections.append(Detection(
                label=label,
                confidence=conf_show,
                box=(x1, y1, x2, y2),
                color_bgr=color_bgr,
                color_hex=color_hex,
                is_pokemon=is_poke,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ----------------------------------------------------------------

    def _classify_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Run the Pokémon CNN on a BGR crop. Returns (name, confidence)."""
        try:
            pd = self._predict_dl
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = pd._transform(pil).unsqueeze(0).to(pd._device)

            import torch
            import torch.nn.functional as F
            with torch.no_grad():
                logits = pd._model(tensor)
                probs  = F.softmax(logits, dim=1)[0]

            top_conf, top_idx = probs.max(dim=0)
            name = pd._class_names[top_idx.item()]
            return name, float(top_conf)
        except Exception:
            return "unknown", 0.0


# ── Utility ─────────────────────────────────────────────────────────

def _clamp_box(
    xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """Clamp bounding box coordinates to frame dimensions."""
    x1 = int(max(0, xyxy[0]))
    y1 = int(max(0, xyxy[1]))
    x2 = int(min(frame_w - 1, xyxy[2]))
    y2 = int(min(frame_h - 1, xyxy[3]))
    return x1, y1, x2, y2


def draw_detections(
    frame_bgr: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """
    Draw bounding boxes + labels onto a BGR frame (in-place).
    Returns the same frame for convenience.
    """
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = det.color_bgr

        # Box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Label text
        label_text = f"{det.label}  {det.confidence*100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Background rect for label
        lbl_y1 = max(y1 - th - baseline - 6, 0)
        cv2.rectangle(
            frame_bgr,
            (x1, lbl_y1),
            (x1 + tw + 8, y1),
            color, -1,
        )

        # Text (white on colored background)
        cv2.putText(
            frame_bgr, label_text,
            (x1 + 4, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame_bgr
