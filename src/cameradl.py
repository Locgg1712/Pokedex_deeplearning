# src/cameradl.py
# Real-time Webcam Pokémon Recognition — CustomTkinter + OpenCV + YOLO + CNN
# Detects MULTIPLE objects per frame with bounding boxes.
# "Person" class is detected directly by YOLOv8n.
# Other regions are classified by the Pokémon CNN.

import cv2
import threading
import time
import numpy as np
from PIL import Image
import customtkinter as ctk

from src.detector import ObjectDetector, draw_detections, Detection


# ==============================
# CAMERA RECOGNITION WINDOW
# ==============================

class CameraWindow(ctk.CTkToplevel):
    """
    Pop-up window that streams webcam frames and runs multi-object
    detection + CNN classification in real-time.

    Detection runs on a background thread (every PRED_INTERVAL seconds)
    to keep the UI smooth even on CPU.
    """

    PRED_INTERVAL = 0.6       # seconds between detection runs
    DISPLAY_SIZE  = (480, 360)  # width × height shown in UI
    CAPTURE_SIZE  = (640, 480)  # resolution requested from camera
    MAX_LIST_ROWS = 8           # max rows in detection list panel

    COLOR_HIGH = "#22C55E"
    COLOR_MID  = "#FACC15"
    COLOR_LOW  = "#FF3E3E"
    COLOR_GREY = "#64748B"

    def __init__(self, master, on_capture_callback=None):
        """
        Args:
            master: Parent CTk window.
            on_capture_callback: Optional callable(image_path) called when
                                  user captures a frame to the main app.
        """
        super().__init__(master)

        self.title("📷 Camera — Multi-Object Recognition")
        self.geometry("900x700")
        self.resizable(False, False)
        self.configure(fg_color="#0F0F14")

        self.on_capture_callback = on_capture_callback

        # State
        self._cap           = None
        self._running       = False
        self._lock          = threading.Lock()
        self._latest_frame  = None          # BGR frame (numpy)
        self._detections    = []            # List[Detection] — latest results
        self._last_pred_t   = 0.0
        self._predicting    = False
        self._paused        = False
        self._frame_count   = 0
        self._fps_t0        = time.time()
        self._latency_ms    = 0

        # Load detector (YOLO + CNN) in a thread to avoid blocking UI open
        self._detector      = None
        self._detector_ready = False
        threading.Thread(target=self._load_detector, daemon=True).start()

        self._build_ui()
        self._start_camera()
        self._poll()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # DETECTOR LOADING
    # ------------------------------------------------------------------

    def _load_detector(self):
        try:
            self._detector = ObjectDetector()
            self._detector_ready = True
            self.after(0, lambda: self._overlay_label.configure(
                text="✅ Detector ready — nhận diện đang chạy",
                text_color="#22C55E"))
        except Exception as e:
            self.after(0, lambda: self._overlay_label.configure(
                text=f"❌ Không tải được detector: {e}",
                text_color="#FF3E3E"))

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Title bar ────────────────────────────────────────────────
        bar = ctk.CTkFrame(self, fg_color="#1A1A24", height=52, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="📷  Real-time Multi-Object Recognition",
                     font=ctk.CTkFont("Arial", 17, "bold"),
                     text_color="#F1F5F9").pack(side="left", padx=18, pady=14)

        self._status_dot = ctk.CTkLabel(bar, text="● LIVE",
                                        font=ctk.CTkFont("Arial", 12, "bold"),
                                        text_color="#22C55E")
        self._status_dot.pack(side="right", padx=18)

        # ── Body: camera left | info right ───────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(10, 0))

        # --- Camera feed ---
        cam_frame = ctk.CTkFrame(body, fg_color="#1A1A24", corner_radius=12)
        cam_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self._cam_label = ctk.CTkLabel(cam_frame, text="",
                                       width=self.DISPLAY_SIZE[0],
                                       height=self.DISPLAY_SIZE[1])
        self._cam_label.pack(padx=8, pady=8)

        self._overlay_label = ctk.CTkLabel(
            cam_frame,
            text="⏳ Đang tải detector (YOLO + CNN)...",
            font=ctk.CTkFont("Arial", 13),
            text_color="#94A3B8")
        self._overlay_label.pack(pady=(0, 8))

        # --- Info panel ---
        info = ctk.CTkFrame(body, fg_color="#1A1A24", corner_radius=12, width=300)
        info.pack(side="right", fill="y")
        info.pack_propagate(False)

        ctk.CTkLabel(info, text="ĐỐI TƯỢNG PHÁT HIỆN",
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color="#94A3B8").pack(pady=(18, 6))

        # Detection count badge
        self._count_label = ctk.CTkLabel(
            info, text="0 đối tượng",
            font=ctk.CTkFont("Arial", 11),
            text_color="#64748B")
        self._count_label.pack(pady=(0, 8))

        ctk.CTkFrame(info, height=1, fg_color="#2D2D3A").pack(fill="x", padx=16, pady=(0, 8))

        # Scrollable detection list
        self._det_scroll = ctk.CTkScrollableFrame(
            info, fg_color="transparent",
            scrollbar_button_color="#3F3F50",
            scrollbar_button_hover_color="#5F5F70",
            height=200)
        self._det_scroll.pack(fill="x", padx=8)

        ctk.CTkFrame(info, height=1, fg_color="#2D2D3A").pack(fill="x", padx=16, pady=10)

        # FPS / latency
        self._fps_label = ctk.CTkLabel(info, text="FPS: —  |  Latency: —",
                                        font=ctk.CTkFont("Arial", 11),
                                        text_color="#475569")
        self._fps_label.pack()

        # Prediction interval slider
        ctk.CTkLabel(info, text="Tần suất nhận diện",
                     font=ctk.CTkFont("Arial", 11),
                     text_color="#64748B").pack(pady=(10, 0))
        self._interval_slider = ctk.CTkSlider(
            info, from_=0.2, to=2.0, number_of_steps=18,
            width=230, command=self._on_interval_change)
        self._interval_slider.set(self.PRED_INTERVAL)
        self._interval_slider.pack(pady=(4, 0))
        self._interval_val_lbl = ctk.CTkLabel(
            info, text=f"{self.PRED_INTERVAL:.1f}s / lần",
            font=ctk.CTkFont("Arial", 11), text_color="#64748B")
        self._interval_val_lbl.pack()

        # Legend
        ctk.CTkFrame(info, height=1, fg_color="#2D2D3A").pack(fill="x", padx=16, pady=10)
        legend_frame = ctk.CTkFrame(info, fg_color="transparent")
        legend_frame.pack(fill="x", padx=12, pady=(0, 6))
        self._add_legend(legend_frame, "●", "#22C55E", "Người (Person)")
        self._add_legend(legend_frame, "●", "#FF3E3E", "Pokémon")
        self._add_legend(legend_frame, "●", "#64748B", "Vật thể khác")

        # ── Bottom buttons ────────────────────────────────────────────
        btns = ctk.CTkFrame(self, fg_color="transparent", height=64)
        btns.pack(fill="x", padx=16, pady=12)
        btns.pack_propagate(False)

        self._capture_btn = ctk.CTkButton(
            btns, text="📸 Chụp & Gửi sang App",
            font=ctk.CTkFont("Arial", 14, "bold"),
            width=230, height=42,
            fg_color="#3B82F6", hover_color="#2563EB",
            corner_radius=10,
            command=self._capture_frame)
        self._capture_btn.pack(side="left", padx=(0, 10))

        self._pause_btn = ctk.CTkButton(
            btns, text="⏸ Tạm dừng",
            font=ctk.CTkFont("Arial", 14),
            width=140, height=42,
            fg_color="#374151", hover_color="#1F2937",
            corner_radius=10,
            command=self._toggle_pause)
        self._pause_btn.pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btns, text="✖ Đóng",
            font=ctk.CTkFont("Arial", 14),
            width=100, height=42,
            fg_color="#7F1D1D", hover_color="#991B1B",
            corner_radius=10,
            command=self._on_close).pack(side="right")

    def _add_legend(self, parent, dot, color, text):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(anchor="w", pady=1)
        ctk.CTkLabel(row, text=dot, text_color=color,
                     font=ctk.CTkFont("Arial", 14, "bold")).pack(side="left")
        ctk.CTkLabel(row, text=f"  {text}",
                     font=ctk.CTkFont("Arial", 11),
                     text_color="#94A3B8").pack(side="left")

    # ------------------------------------------------------------------
    # CAMERA THREAD
    # ------------------------------------------------------------------

    def _start_camera(self, index=0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(1)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAPTURE_SIZE[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_SIZE[1])
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._running = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
        else:
            self._overlay_label.configure(text="❌ Không tìm thấy camera!")
            self._status_dot.configure(text="● OFFLINE", text_color="#FF3E3E")

    def _capture_loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            with self._lock:
                self._latest_frame = frame.copy()

            # Trigger detection on interval (only when detector is ready)
            now = time.time()
            if (self._detector_ready
                    and not self._predicting
                    and now - self._last_pred_t >= self.PRED_INTERVAL):
                self._predicting  = True
                self._last_pred_t = now
                threading.Thread(
                    target=self._run_detection,
                    args=(frame.copy(),),
                    daemon=True).start()

            time.sleep(0.01)

    # ------------------------------------------------------------------
    # DETECTION THREAD
    # ------------------------------------------------------------------

    def _run_detection(self, frame_bgr: np.ndarray):
        """Run YOLO + CNN detection on a single frame (background thread)."""
        t0 = time.time()
        try:
            dets = self._detector.detect(frame_bgr)
            self._latency_ms = int((time.time() - t0) * 1000)
            with self._lock:
                self._detections = dets
        except Exception as e:
            print(f"[camera] Detection error: {e}")
        finally:
            self._predicting = False

    # ------------------------------------------------------------------
    # UI UPDATE LOOP (~33 fps, main thread)
    # ------------------------------------------------------------------

    def _poll(self):
        if not self._running:
            return

        with self._lock:
            frame     = self._latest_frame
            dets      = list(self._detections)

        if frame is not None and not self._paused:
            self._frame_count += 1
            self._update_display(frame, dets)

        # FPS counter every 60 frames
        if self._frame_count % 60 == 0 and self._frame_count > 0:
            elapsed = time.time() - self._fps_t0
            fps = 60 / max(elapsed, 0.001)
            self._fps_t0 = time.time()
            self._fps_label.configure(
                text=f"FPS: {fps:.0f}  |  Latency: {self._latency_ms} ms")

        self.after(30, self._poll)

    def _update_display(self, frame_bgr: np.ndarray, dets: list):
        """Render bounding boxes and update info panel."""
        display = cv2.resize(frame_bgr, self.DISPLAY_SIZE)

        # Draw all bounding boxes (scale coords to display size)
        scale_x = self.DISPLAY_SIZE[0] / frame_bgr.shape[1]
        scale_y = self.DISPLAY_SIZE[1] / frame_bgr.shape[0]

        scaled_dets = []
        for d in dets:
            x1, y1, x2, y2 = d.box
            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x)
            sy2 = int(y2 * scale_y)
            from dataclasses import replace
            scaled_dets.append(replace(d, box=(sx1, sy1, sx2, sy2)))

        draw_detections(display, scaled_dets)

        # LIVE indicator dot
        cv2.circle(display, (self.DISPLAY_SIZE[0] - 18, 18), 8, (0, 230, 80), -1)
        cv2.putText(display, "LIVE",
                    (self.DISPLAY_SIZE[0] - 60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 255, 200), 1, cv2.LINE_AA)

        # Push to label
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil     = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(pil, size=self.DISPLAY_SIZE)
        self._cam_label.configure(image=ctk_img)
        self._cam_label.image = ctk_img  # keep reference

        # Update info panel
        self._update_detection_list(dets)

    def _update_detection_list(self, dets: list):
        """Rebuild the detection list in the right panel."""
        self._count_label.configure(
            text=f"{len(dets)} đối tượng" if dets else "Không phát hiện gì")

        # Clear old rows
        for w in self._det_scroll.winfo_children():
            w.destroy()

        if not dets:
            ctk.CTkLabel(self._det_scroll,
                         text="Đưa đối tượng vào khung hình...",
                         font=ctk.CTkFont("Arial", 11),
                         text_color="#475569").pack(pady=12)
            return

        for i, det in enumerate(dets[:self.MAX_LIST_ROWS]):
            self._create_det_row(i + 1, det)

        if len(dets) > self.MAX_LIST_ROWS:
            ctk.CTkLabel(self._det_scroll,
                         text=f"... và {len(dets) - self.MAX_LIST_ROWS} đối tượng khác",
                         font=ctk.CTkFont("Arial", 10),
                         text_color="#475569").pack(pady=(4, 0))

    def _create_det_row(self, rank: int, det: Detection):
        """Create one row card in the detection list."""
        card = ctk.CTkFrame(self._det_scroll, fg_color="#22222E", corner_radius=8, height=48)
        card.pack(fill="x", pady=3, padx=2)
        card.pack_propagate(False)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=6)

        # Rank + label
        ctk.CTkLabel(inner,
                     text=f"{rank}. {det.label}",
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color=det.color_hex).pack(side="left")

        # Confidence on the right
        ctk.CTkLabel(inner,
                     text=f"{det.confidence*100:.0f}%",
                     font=ctk.CTkFont("Arial", 12, "bold"),
                     text_color=det.color_hex).pack(side="right")

    # ------------------------------------------------------------------
    # BUTTONS
    # ------------------------------------------------------------------

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._pause_btn.configure(text="▶ Tiếp tục")
            self._status_dot.configure(text="● PAUSED", text_color="#FACC15")
            self._overlay_label.configure(text="Đã tạm dừng", text_color="#94A3B8")
        else:
            self._pause_btn.configure(text="⏸ Tạm dừng")
            self._status_dot.configure(text="● LIVE", text_color="#22C55E")

    def _capture_frame(self):
        """Save current frame and pass top Pokémon detection to main app."""
        with self._lock:
            frame = self._latest_frame
            dets  = list(self._detections)

        if frame is None:
            return

        import os, tempfile
        tmp = tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False,
            dir=tempfile.gettempdir(), prefix="pokedex_capture_")
        tmp.close()
        cv2.imwrite(tmp.name, frame)

        self._capture_btn.configure(text="✅ Đã gửi!", fg_color="#16A34A")
        self.after(1500, lambda: self._capture_btn.configure(
            text="📸 Chụp & Gửi sang App", fg_color="#3B82F6"))

        if self.on_capture_callback:
            # Send the best Pokémon detection path (or raw frame if none)
            self.on_capture_callback(tmp.name)

    def _on_interval_change(self, val):
        self.PRED_INTERVAL = round(float(val), 1)
        self._interval_val_lbl.configure(text=f"{self.PRED_INTERVAL:.1f}s / lần")

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def _on_close(self):
        self._running = False
        time.sleep(0.05)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self.destroy()