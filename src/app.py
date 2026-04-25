# src/app.py
# Pokédex Deep Learning GUI — CustomTkinter application.
# Pipeline: Image → CNN → Label → PokéAPI → History → UI

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import threading

from src.predict_dl import predict
from src.preprocess import extract_pokemon_debug
from src.api import fetch_pokemon_info
from src.history import log_prediction, get_history

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class PokedexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pokédex DL — Deep Learning")
        self.geometry("1200x820")
        self.minsize(1000, 720)

        # ===== Color Palette =====
        self.bg_color       = "#0F0F14"
        self.panel_bg       = "#1A1A24"
        self.card_bg        = "#22222E"
        self.accent_red     = "#FF3E3E"
        self.accent_blue    = "#3B82F6"
        self.accent_green   = "#22C55E"
        self.accent_yellow  = "#FACC15"
        self.text_primary   = "#F1F5F9"
        self.text_secondary = "#94A3B8"
        self.border_color   = "#2D2D3A"

        self.configure(fg_color=self.bg_color)

        # Layout: 3 columns — History | Main | Info
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=260)  # History
        self.grid_columnconfigure(1, weight=1, minsize=440)  # Main
        self.grid_columnconfigure(2, weight=0, minsize=280)  # Info

        self._build_history_panel()
        self._build_main_panel()
        self._build_info_panel()

        # Pokéball animation state
        self.pokeball_state = "closed"
        self.top_rely = 0.25
        self.bot_rely = 0.75

        # Load history on start
        self._refresh_history()

    # ===========================================================
    #  HISTORY PANEL (Left)
    # ===========================================================

    def _build_history_panel(self):
        self.history_panel = ctk.CTkFrame(self, fg_color=self.panel_bg, corner_radius=0,
                                          border_width=1, border_color=self.border_color)
        self.history_panel.grid(row=0, column=0, sticky="nsew")

        # Title
        header = ctk.CTkFrame(self.history_panel, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(20, 8))

        ctk.CTkLabel(header, text="📋 LỊCH SỬ",
                     font=ctk.CTkFont("Arial", 16, "bold"),
                     text_color=self.text_primary).pack(side="left")

        ctk.CTkButton(header, text="Xóa", width=50, height=28,
                      fg_color="#3F3F50", hover_color=self.accent_red,
                      font=ctk.CTkFont(size=12), corner_radius=6,
                      command=self._clear_history).pack(side="right")

        # Scrollable list
        self.history_scroll = ctk.CTkScrollableFrame(
            self.history_panel, fg_color="transparent",
            scrollbar_button_color="#3F3F50",
            scrollbar_button_hover_color="#5F5F70")
        self.history_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 12))

    def _refresh_history(self):
        """Reload history list from database."""
        for widget in self.history_scroll.winfo_children():
            widget.destroy()

        records = get_history(limit=30)

        if not records:
            ctk.CTkLabel(self.history_scroll, text="Chưa có dự đoán nào",
                         text_color=self.text_secondary,
                         font=ctk.CTkFont(size=12)).pack(pady=30)
            return

        for rec in records:
            self._create_history_card(rec)

    def _create_history_card(self, rec):
        """Create a single history entry card."""
        card = ctk.CTkFrame(self.history_scroll, fg_color=self.card_bg,
                            corner_radius=8, height=60)
        card.pack(fill="x", pady=3, padx=4)
        card.pack_propagate(False)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=6)

        # Name + confidence
        name_lbl = ctk.CTkLabel(inner, text=rec["predicted_label"].capitalize(),
                                font=ctk.CTkFont(size=13, weight="bold"),
                                text_color=self.text_primary)
        name_lbl.pack(anchor="w")

        conf = rec["confidence"]
        color = self.accent_green if conf > 0.8 else (self.accent_yellow if conf > 0.5 else self.accent_red)

        detail_text = f"{conf*100:.1f}%  •  {rec['timestamp'][:16]}"
        ctk.CTkLabel(inner, text=detail_text,
                     font=ctk.CTkFont(size=11), text_color=color).pack(anchor="w")

    def _clear_history(self):
        from src.history import clear_history
        clear_history()
        self._refresh_history()

    # ===========================================================
    #  MAIN PANEL (Center) — Pokéball + Prediction
    # ===========================================================

    def _build_main_panel(self):
        self.main_panel = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=0)
        self.main_panel.grid(row=0, column=1, sticky="nsew")

        # Top header area
        top = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 0))

        ctk.CTkLabel(top, text="🔴 Pokédex DL",
                     font=ctk.CTkFont("Arial", 28, "bold"),
                     text_color=self.accent_red).pack(side="left")

        ctk.CTkLabel(top, text="Deep Learning CNN",
                     font=ctk.CTkFont("Arial", 13),
                     text_color=self.text_secondary).pack(side="left", padx=12)

        # Upload button
        self.upload_btn = ctk.CTkButton(
            top, text="📁 Chọn Ảnh", width=130, height=36,
            fg_color=self.accent_red, hover_color="#D93232",
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=8, command=self.load_image)
        self.upload_btn.pack(side="right")

        # ===== Pokéball container =====
        self.pokeball_container = ctk.CTkFrame(self.main_panel, width=520, height=520,
                                                fg_color="transparent")
        self.pokeball_container.pack(expand=True)

        # Inner content (shown when Pokéball opens)
        self.inner_frame = ctk.CTkFrame(self.pokeball_container, width=500, height=500,
                                        fg_color=self.card_bg, corner_radius=20)
        self.inner_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.inner_frame.pack_propagate(False)

        # --- 2x2 Image Grid ---
        self.grid_frame = ctk.CTkFrame(self.inner_frame, fg_color="transparent")
        self.grid_frame.pack(pady=(20, 10))
        self.grid_frame.grid_columnconfigure((0, 1), weight=1)
        self.grid_frame.grid_rowconfigure((0, 1), weight=1)
        
        self.panel_original = self._create_image_panel(self.grid_frame, "Original", 0, 0)
        self.panel_blur     = self._create_image_panel(self.grid_frame, "DL Denoised", 0, 1)
        self.panel_edge     = self._create_image_panel(self.grid_frame, "Edges", 1, 0)
        self.panel_final    = self._create_image_panel(self.grid_frame, "Final", 1, 1)

        # --- Prediction result ---
        self.pokemon_name_label = ctk.CTkLabel(
            self.inner_frame, text="Chưa xác định",
            font=ctk.CTkFont("Arial", 32, "bold"),
            text_color=self.text_primary)
        self.pokemon_name_label.pack(pady=(5, 2))

        # Confidence bar
        bar_frame = ctk.CTkFrame(self.inner_frame, fg_color="transparent")
        bar_frame.pack(pady=(5, 2))

        self.confidence_bar = ctk.CTkProgressBar(bar_frame, width=260, height=12,
                                                  progress_color=self.text_secondary,
                                                  fg_color="#1A1A24")
        self.confidence_bar.set(0)
        self.confidence_bar.pack(side="left")

        self.confidence_label = ctk.CTkLabel(bar_frame, text="0%",
                                              font=ctk.CTkFont(size=13, weight="bold"),
                                              text_color=self.text_secondary, width=55)
        self.confidence_label.pack(side="left", padx=(8, 0))

        # Top-3 predictions
        self.top3_label = ctk.CTkLabel(self.inner_frame, text="",
                                        font=ctk.CTkFont("Arial", 12),
                                        text_color=self.text_secondary, justify="center")
        self.top3_label.pack(pady=(8, 0))

        # ===== Pokéball shell =====
        pokeball_red   = "#e74c3c"
        pokeball_white = "#ffffff"
        pokeball_black = "#000000"

        self.top_half = ctk.CTkFrame(self.pokeball_container, width=520, height=260,
                                     fg_color=pokeball_red, corner_radius=30)
        self.top_half.place(relx=0.5, rely=0.25, anchor="center")

        self.bot_half = ctk.CTkFrame(self.pokeball_container, width=520, height=260,
                                     fg_color=pokeball_white, corner_radius=30)
        self.bot_half.place(relx=0.5, rely=0.75, anchor="center")

        # Divider lines
        self.top_line = ctk.CTkFrame(self.top_half, width=520, height=10,
                                     fg_color=pokeball_black)
        self.top_line.place(relx=0.5, rely=1.0, anchor="s")

        self.bot_line = ctk.CTkFrame(self.bot_half, width=520, height=10,
                                     fg_color=pokeball_black)
        self.bot_line.place(relx=0.5, rely=0.0, anchor="n")

        # Center button
        self.center_ring = ctk.CTkFrame(self.pokeball_container, width=120, height=120,
                                        corner_radius=60, fg_color=pokeball_black)
        self.center_ring.place(relx=0.5, rely=0.5, anchor="center")

        self.center_btn_white = ctk.CTkFrame(self.center_ring, width=88, height=88,
                                             corner_radius=44, fg_color="#F0F0F0")
        self.center_btn_white.place(relx=0.5, rely=0.5, anchor="center")

        self.center_btn = ctk.CTkButton(
            self.center_btn_white, width=64, height=64, corner_radius=32,
            fg_color="#FFFFFF", hover_color="#DDDDDD", text="",
            border_width=2, border_color="#CCCCCC",
            command=self.load_image)
        self.center_btn.place(relx=0.5, rely=0.5, anchor="center")

        # Idle hint
        self.idle_label = ctk.CTkLabel(self.bot_half, text="Click để phân tích",
                                       font=ctk.CTkFont("Arial", 15, "bold"),
                                       text_color="#A0A0A0")
        self.idle_label.place(relx=0.5, rely=0.6, anchor="center")

    def _create_image_panel(self, parent, title, row, col):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, padx=8, pady=8)
        
        lbl_title = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color=self.text_secondary)
        lbl_title.pack(pady=(0, 2))

        lbl_img = ctk.CTkLabel(frame, text="", image=self._placeholder(150, 150))
        lbl_img.pack()
        return lbl_img

    # ===========================================================
    #  INFO PANEL (Right) — PokéAPI Data
    # ===========================================================

    def _build_info_panel(self):
        self.info_panel = ctk.CTkFrame(self, fg_color=self.panel_bg, corner_radius=0,
                                       border_width=1, border_color=self.border_color)
        self.info_panel.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(self.info_panel, text="📖 THÔNG TIN POKÉMON",
                     font=ctk.CTkFont("Arial", 16, "bold"),
                     text_color=self.text_primary).pack(padx=16, pady=(20, 12), anchor="w")

        # Info container
        self.info_container = ctk.CTkFrame(self.info_panel, fg_color=self.card_bg,
                                           corner_radius=12)
        self.info_container.pack(fill="x", padx=12, pady=(0, 8))

        # Sprite image
        self.sprite_label = ctk.CTkLabel(self.info_container, text="",
                                          image=self._placeholder(140, 140))
        self.sprite_label.pack(pady=(16, 8))

        # Info fields
        self.info_name = self._info_row(self.info_container, "Tên", "—")
        self.info_types = self._info_row(self.info_container, "Loại", "—")
        self.info_height = self._info_row(self.info_container, "Chiều cao", "—")
        self.info_weight = self._info_row(self.info_container, "Cân nặng", "—")

        # Separator
        ctk.CTkFrame(self.info_panel, height=1, fg_color=self.border_color).pack(fill="x", padx=16, pady=12)

        # Status label
        self.status_label = ctk.CTkLabel(self.info_panel, text="⏳ Sẵn sàng phân tích",
                                          font=ctk.CTkFont(size=12),
                                          text_color=self.text_secondary, wraplength=240)
        self.status_label.pack(padx=16, pady=(0, 8), anchor="w")

        # Type color badges area
        self.type_badge_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.type_badge_frame.pack(fill="x", padx=16, pady=(0, 16))

    def _info_row(self, parent, label_text, default_value):
        """Create a labeled info row and return the value label for updating."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(row, text=label_text,
                     font=ctk.CTkFont(size=12),
                     text_color=self.text_secondary, width=70, anchor="w").pack(side="left")

        val = ctk.CTkLabel(row, text=default_value,
                           font=ctk.CTkFont(size=13, weight="bold"),
                           text_color=self.text_primary, anchor="w")
        val.pack(side="left", padx=(4, 0))
        return val

    # ===========================================================
    #  POKÉBALL ANIMATION
    # ===========================================================

    def _set_pokeball_closed(self):
        self.top_half.place(relx=0.5, rely=0.25, anchor="center")
        self.bot_half.place(relx=0.5, rely=0.75, anchor="center")
        self.center_ring.place(relx=0.5, rely=0.5, anchor="center")
        self.idle_label.place(relx=0.5, rely=0.6, anchor="center")
        self.pokeball_state = "closed"
        self.top_rely = 0.25
        self.bot_rely = 0.75

    def _open_animation(self, step=0):
        if step == 0:
            self.pokeball_state = "opening"
            self.center_ring.place_forget()
            self.idle_label.place_forget()

        if step <= 25:
            self.top_rely = 0.25 - (0.48 * (step / 25))
            self.bot_rely = 0.75 + (0.48 * (step / 25))
            self.top_half.place(relx=0.5, rely=self.top_rely, anchor="center")
            self.bot_half.place(relx=0.5, rely=self.bot_rely, anchor="center")
            self.after(12, lambda: self._open_animation(step + 1))
        else:
            self.pokeball_state = "open"

    # ===========================================================
    #  HELPERS
    # ===========================================================

    def _placeholder(self, w, h):
        img = Image.new("RGB", (w, h), self.card_bg)
        return ctk.CTkImage(img, size=(w, h))

    def _cv2_to_ctk(self, img, size=(280, 280)):
        if img is None:
            return self._placeholder(*size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=size)

    # ===========================================================
    #  TYPE COLORS (for badges)
    # ===========================================================

    TYPE_COLORS = {
        "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0",
        "Electric": "#F7D02C", "Grass": "#7AC74C", "Ice": "#96D9D6",
        "Fighting": "#C22E28", "Poison": "#A33EA1", "Ground": "#E2BF65",
        "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
        "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC",
        "Dark": "#705746", "Steel": "#B7B7CE", "Fairy": "#D685AD",
    }

    def _show_type_badges(self, types):
        """Display colored type badges."""
        for w in self.type_badge_frame.winfo_children():
            w.destroy()

        for t in types:
            color = self.TYPE_COLORS.get(t, "#666666")
            badge = ctk.CTkLabel(self.type_badge_frame, text=f" {t} ",
                                 fg_color=color, corner_radius=6,
                                 font=ctk.CTkFont(size=12, weight="bold"),
                                 text_color="#FFFFFF", height=26)
            badge.pack(side="left", padx=(0, 6), pady=2)

    # ===========================================================
    #  MAIN ACTION: LOAD IMAGE → PREDICT → API → HISTORY
    # ===========================================================

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh Pokémon",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        # Close Pokéball while processing
        self._set_pokeball_closed()
        self.update()

        self.status_label.configure(text="⏳ Đang phân tích...")

        # Fetch 4-stage pipeline images
        steps = extract_pokemon_debug(path)
        if steps:
            self.panel_original.configure(image=self._cv2_to_ctk(steps["original"], (150, 150)))
            self.panel_blur.configure(image=self._cv2_to_ctk(steps["blur"], (150, 150)))
            self.panel_edge.configure(image=self._cv2_to_ctk(steps["edges"], (150, 150)))
            self.panel_final.configure(image=self._cv2_to_ctk(steps["combine"], (150, 150)))

        # --- CNN Prediction ---
        try:
            name, conf, top3 = predict(path)

            self.pokemon_name_label.configure(text=name.capitalize(),
                                              text_color=self.text_primary)
            self.confidence_bar.set(conf)
            self.confidence_label.configure(text=f"{conf*100:.1f}%")

            if conf > 0.8:
                bar_color = self.accent_green
            elif conf > 0.5:
                bar_color = self.accent_yellow
            else:
                bar_color = self.accent_red
            self.confidence_bar.configure(progress_color=bar_color)
            self.confidence_label.configure(text_color=bar_color)

            # Top-3
            if top3:
                lines = [f"{n.capitalize()}: {p*100:.1f}%" for n, p in top3]
                self.top3_label.configure(text="Top-3:  " + "  |  ".join(lines))

            # --- Log to History ---
            log_prediction(path, name, conf)
            self._refresh_history()

            # --- Fetch PokéAPI in background ---
            self.status_label.configure(text="🌐 Đang tải thông tin từ PokéAPI...")
            threading.Thread(target=self._fetch_api, args=(name,), daemon=True).start()

        except Exception as e:
            self.pokemon_name_label.configure(text="Lỗi Model", text_color=self.accent_red)
            self.confidence_bar.set(0)
            self.confidence_label.configure(text="0%")
            self.top3_label.configure(text="")
            self.status_label.configure(text=f"❌ {str(e)}")

        # Open Pokéball animation
        self.after(200, self._open_animation)

    def _fetch_api(self, pokemon_name):
        """Fetch PokéAPI data on a background thread, then update UI on main thread."""
        info = fetch_pokemon_info(pokemon_name)
        self.after(0, lambda: self._update_info_panel(info))

    def _update_info_panel(self, info):
        """Update the info panel with PokéAPI data (called on main thread)."""
        if info is None:
            self.status_label.configure(text="⚠️ Không thể kết nối PokéAPI")
            self.info_name.configure(text="—")
            self.info_types.configure(text="—")
            self.info_height.configure(text="—")
            self.info_weight.configure(text="—")
            self._show_type_badges([])
            return

        self.info_name.configure(text=info["name"])
        self.info_types.configure(text=" / ".join(info["types"]))
        self.info_height.configure(text=f"{info['height']} m")
        self.info_weight.configure(text=f"{info['weight']} kg")

        self._show_type_badges(info["types"])
        self.status_label.configure(text="✅ Phân tích hoàn tất")

        # Load sprite from URL
        if info.get("sprite"):
            threading.Thread(target=self._load_sprite, args=(info["sprite"],),
                             daemon=True).start()

    def _load_sprite(self, url):
        """Download and display the official artwork sprite."""
        try:
            import requests
            from io import BytesIO
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGBA")

            # Composite on dark background
            bg = Image.new("RGBA", img.size, self.card_bg + "FF")
            bg.paste(img, (0, 0), img)
            final = bg.convert("RGB")

            ctk_img = ctk.CTkImage(light_image=final, dark_image=final, size=(140, 140))
            self.after(0, lambda: self.sprite_label.configure(image=ctk_img))
        except Exception:
            pass  # Silently fail — sprite is non-critical


if __name__ == "__main__":
    app = PokedexApp()
    app.mainloop()