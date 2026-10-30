import os
import sys
import time
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import psutil

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    matplotlib = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.edge import EdgePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
SEQ_LEN = 60
FORECAST_HORIZON = 30
WINDOW_DISPLAY = 200

DARK_BG = "#0a0b10"
PANEL_BG = "#12141d"
PANEL_BG2 = "#161a28"
ACCENT = "#00f2ff"
TEXT = "#e8eaf6"
MUTED = "#8a93a8"
OK_COLOR = "#00ff88"
WARN_COLOR = "#ffaa00"
BAD_COLOR = "#ff4444"


def list_input_files():
    if not os.path.isdir(INPUT_DIR):
        return []
    return [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".csv")]


def ensure_default_input():
    os.makedirs(INPUT_DIR, exist_ok=True)
    if not list_input_files():
        src = os.path.join(BASE_DIR, "household_data_1min_singleindex.csv")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(INPUT_DIR, "household_data_1min_singleindex.csv"))


def display_name(filename):
    return os.path.splitext(filename)[0]


def detect_target_columns(csv_path):
    """Return candidate target columns: non-timestamp numeric columns, grid_import first."""
    try:
        df_head = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return []
    cols = list(df_head.columns)
    non_ts = [c for c in cols if "timestamp" not in c.lower() and "time" not in c.lower()]
    grid = [c for c in non_ts if "grid_import" in c.lower()]
    ordered = grid + [c for c in non_ts if c not in grid]
    return ordered


class DataStream:
    def __init__(self, csv_path, column):
        self.csv_path = csv_path
        self.column = column
        df = pd.read_csv(csv_path, usecols=[column])
        series = df[column].ffill().bfill().fillna(0.0).values.astype(np.float32)
        start = self._first_significant(series)
        self.idx = start + SEQ_LEN
        self.series = series
        self.len = len(series)

    @staticmethod
    def _first_significant(series, threshold_ratio=0.05):
        mx = float(series.max()) if len(series) else 0.0
        if mx <= 0:
            return 0
        thr = mx * threshold_ratio
        for i, v in enumerate(series):
            if v > thr:
                return i
        return 0

    def current_window(self):
        i = self.idx
        return self.series[i - SEQ_LEN:i].copy()

    def current_true(self):
        i = self.idx
        if i >= self.len:
            return None
        return float(self.series[i])

    def advance(self):
        self.idx += 1

    def remaining(self):
        return max(0, self.len - self.idx)


def base_forecast(session, input_name, mn, mx, window, steps):
    w = np.asarray(window, dtype=np.float32).copy()
    mn = float(mn); mx = float(mx)
    out = []
    for _ in range(steps):
        x = ((w - mn) / (mx - mn + 1e-8)).reshape(1, 60, 1).astype(np.float32)
        pred, _ = session.run(None, {input_name: x})
        fp = float(pred[0, 0]) * (mx - mn + 1e-8) + mn
        out.append(fp)
        w = np.roll(w, -1)
        w[-1] = fp
    return out


class SimEngine(threading.Thread):
    def __init__(self, csv_path, column, out_queue, speed_ms=40, memory_path_prefix=None):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.column = column
        self.out_queue = out_queue
        self.speed_ms = speed_ms
        self.memory_path_prefix = memory_path_prefix
        self.running = threading.Event()
        self.paused = threading.Event()
        self.predictor = None
        self.stream = None
        self.stopped = threading.Event()

    def start_sim(self):
        self.running.set()
        self.paused.clear()
        self.stopped.clear()
        super().start()

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def stop(self):
        self.stopped.set()
        self.running.clear()
        self.paused.clear()

    def run(self):
        try:
            self.predictor = EdgePredictor(memory_path_prefix=self.memory_path_prefix)
        except Exception as e:
            self.out_queue.put(("error", f"Failed to load edge model: {e}"))
            return

        try:
            self.stream = DataStream(self.csv_path, self.column)
        except Exception as e:
            self.out_queue.put(("error", f"Failed to load data: {e}"))
            return

        preds = []
        trues = []
        step = 0

        while self.running.is_set() and not self.stopped.is_set():
            if self.paused.is_set():
                time.sleep(0.05)
                continue

            window = self.stream.current_window()
            y_true = self.stream.current_true()
            if y_true is None:
                self.out_queue.put(("done", None))
                break

            t_start = time.time()
            pred, _ = self.predictor.predict(window, y_curr=y_true)
            latency_ms = (time.time() - t_start) * 1000.0

            self.stream.advance()

            preds.append(pred)
            trues.append(y_true)
            step += 1

            if len(preds) > WINDOW_DISPLAY:
                preds = preds[-WINDOW_DISPLAY:]
                trues = trues[-WINDOW_DISPLAY:]

            f_window = self.stream.current_window().copy()
            forecast = base_forecast(self.predictor.session, self.predictor.input_name,
                                     self.predictor.mn, self.predictor.mx,
                                     f_window, FORECAST_HORIZON)

            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            proc = psutil.Process()
            proc_mem_mb = proc.memory_info().rss / (1024 * 1024)
            try:
                disk = psutil.disk_usage(BASE_DIR).percent
            except Exception:
                disk = 0.0

            err = abs(pred - y_true)

            payload = {
                "step": step,
                "pred": pred,
                "true": y_true,
                "err": err,
                "preds": list(preds),
                "trues": list(trues),
                "latency_ms": latency_ms,
                "cpu": cpu,
                "mem": mem,
                "proc_mem_mb": proc_mem_mb,
                "disk": disk,
                "forecast": forecast,
                "remaining": self.stream.remaining(),
                "idx": self.stream.idx,
                "column": self.column,
                "csv": self.csv_path,
            }
            self.out_queue.put(("tick", payload))

            # advance one time-step (1 minute) for the next tick
            self.stream.idx += 1

            time.sleep(self.speed_ms / 1000.0)


class Gauge(tk.Canvas):
    def __init__(self, parent, title, unit, maxval=100.0, warn=80.0, bad=95.0,
                 width=200, height=140, accent=ACCENT, **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                         bg=PANEL_BG, **kwargs)
        self.title = title
        self.unit = unit
        self.maxval = maxval
        self.warn = warn
        self.bad = bad
        self.accent = accent
        self.value = 0.0
        self.width = width
        self.height = height
        self.cx = width / 2
        self.cy = height * 0.62
        self.r = min(width, height) * 0.30
        self._draw_static()
        self.needle = None
        self.val_text = None
        self.title_text = None
        self._draw()

    def _draw_static(self):
        self.create_text(self.cx, self.height * 0.16, text=self.title,
                         fill=MUTED, font=("Consolas", 10, "bold"), anchor="center")
        self.create_arc(self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r,
                        start=225, extent=270, style=tk.ARC, outline="#222a3a", width=8)
        zones = [(0, self.warn, "#22303e"), (self.warn, self.bad, "#3a2a10"), (self.bad, self.maxval, "#3a1010")]
        for lo, hi, col in zones:
            span = (hi - lo) / self.maxval * 270.0
            start = 225 + (lo / self.maxval * 270.0)
            self.create_arc(self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r,
                            start=start, extent=span, style=tk.ARC, outline=col, width=8)

    def set(self, value):
        self.value = max(0.0, min(self.maxval, float(value)))
        self._draw()

    def _draw(self):
        if self.needle is not None:
            self.delete(self.needle)
        if self.val_text is not None:
            self.delete(self.val_text)
        if self.title_text is not None:
            self.delete(self.title_text)

        ang = 225 + (self.value / self.maxval) * 270.0
        rad = math.radians(ang)
        x2 = self.cx + self.r * 0.82 * math.cos(rad)
        y2 = self.cy - self.r * 0.82 * math.sin(rad)
        self.needle = self.create_line(self.cx, self.cy, x2, y2, fill=self.accent, width=3, capstyle=tk.ROUND)

        if self.value >= self.bad:
            vcol = BAD_COLOR
        elif self.value >= self.warn:
            vcol = WARN_COLOR
        else:
            vcol = OK_COLOR
        self.val_text = self.create_text(self.cx, self.cy + self.r * 0.28,
                                         text=f"{self.value:.1f}{self.unit}", fill=vcol,
                                         font=("Consolas", 16, "bold"), anchor="center")
        self.title_text = self.create_text(self.cx, self.height * 0.16, text=self.title,
                                           fill=MUTED, font=("Consolas", 10, "bold"), anchor="center")


class MetricCard(tk.Frame):
    def __init__(self, parent, label, value_fn, fmt="{:.2f}", color=ACCENT, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.value_fn = value_fn
        self.fmt = fmt
        self.color = color
        self.lbl = tk.Label(self, text=label, bg=PANEL_BG, fg=MUTED,
                            font=("Consolas", 9, "bold"))
        self.lbl.pack(anchor="w", padx=12, pady=(10, 0))
        self.val = tk.Label(self, text="--", bg=PANEL_BG, fg=color,
                            font=("Consolas", 20, "bold"))
        self.val.pack(anchor="w", padx=12, pady=(0, 10))
        self.configure(highlightbackground="#222a3a", highlightthickness=1, bd=0)

    def update(self, payload):
        v = self.value_fn(payload)
        self.val.config(text=self.fmt.format(v))

    def pulse(self):
        self.val.config(fg="#ffffff")
        self.after(120, lambda: self.val.config(fg=self.color))


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("simulator")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("1300x820")
        try:
            self.root.wm_attributes("-alpha", 0.985)
        except Exception:
            pass

        self.q = queue.Queue()
        self.engine = None
        self.speed_var = tk.DoubleVar(value=40.0)

        self._build_style()
        self._build_ui()

        self.poll_id = None
        self.poll()

    def _build_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=PANEL_BG, background=PANEL_BG, foreground=TEXT)

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=PANEL_BG, height=56)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="simulator", bg=PANEL_BG, fg=ACCENT,
                 font=("Consolas", 18, "bold")).pack(side=tk.LEFT, padx=16, pady=14)
        tk.Label(header, text="Raspberry Pi 5  •  ONNX Runtime  •  CNN-GRU + HNSW Memory",
                 bg=PANEL_BG, fg=MUTED, font=("Consolas", 9)).pack(side=tk.LEFT, padx=8, pady=14)

        # Controls bar
        ctrl = tk.Frame(self.root, bg=PANEL_BG2, height=62)
        ctrl.pack(side=tk.TOP, fill=tk.X)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="INPUT DATA (input/ folder)", bg=PANEL_BG2, fg=MUTED, font=("Consolas", 9, "bold")).pack(side=tk.LEFT, padx=(14, 4), pady=20)
        self.input_files = list_input_files()
        input_labels = [display_name(f) for f in self.input_files] or ["(no inputs)"]
        self.input_var = tk.StringVar(value=input_labels[0] if input_labels else "")
        self.input_combo = ttk.Combobox(ctrl, values=input_labels, state="readonly",
                                        textvariable=self.input_var, width=26)
        self.input_combo.pack(side=tk.LEFT, padx=4, pady=20)
        if input_labels:
            self.input_combo.current(0)
        self.input_combo.bind("<<ComboboxSelected>>", self._on_input_change)

        tk.Label(ctrl, text="TARGET COLUMN", bg=PANEL_BG2, fg=MUTED, font=("Consolas", 9, "bold")).pack(side=tk.LEFT, padx=(18, 4), pady=20)
        self.column_var = tk.StringVar(value="")
        self.column_combo = ttk.Combobox(ctrl, values=[], state="readonly",
                                         textvariable=self.column_var, width=24)
        self.column_combo.pack(side=tk.LEFT, padx=4, pady=20)
        self._refresh_columns()

        tk.Label(ctrl, text="SPEED (ms/tick)", bg=PANEL_BG2, fg=MUTED, font=("Consolas", 9, "bold")).pack(side=tk.LEFT, padx=(18, 4), pady=20)
        sp = tk.Scale(ctrl, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.speed_var,
                      bg=PANEL_BG2, fg=TEXT, highlightthickness=0, troughcolor="#222a3a",
                      activebackground=ACCENT, length=140, showvalue=False)
        sp.pack(side=tk.LEFT, padx=4, pady=16)

        self.btn_start = tk.Button(ctrl, text="▶ START", bg=ACCENT, fg="#000",
                                   font=("Consolas", 10, "bold"), padx=16, pady=4,
                                   relief=tk.FLAT, command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=10, pady=16)
        self.btn_pause = tk.Button(ctrl, text="⏸ PAUSE", bg="#222a3a", fg=TEXT,
                                   font=("Consolas", 10, "bold"), padx=16, pady=4,
                                   relief=tk.FLAT, command=self.pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=4, pady=16)
        self.btn_stop = tk.Button(ctrl, text="⏹ STOP", bg="#3a1010", fg="#ff8888",
                                  font=("Consolas", 10, "bold"), padx=16, pady=4,
                                  relief=tk.FLAT, command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4, pady=16)

        self.status_lbl = tk.Label(ctrl, text="IDLE", bg=PANEL_BG2, fg=MUTED,
                                   font=("Consolas", 9, "bold"))
        self.status_lbl.pack(side=tk.RIGHT, padx=16, pady=20)

        # Main body
        body = tk.Frame(self.root, bg=DARK_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left column: system resources only
        left = tk.Frame(body, bg=PANEL_BG, width=330)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left.pack_propagate(False)

        sec2 = tk.Frame(left, bg=PANEL_BG)
        sec2.pack(fill=tk.X, padx=10, pady=(10, 6))
        tk.Label(sec2, text="EDGE RESOURCES", bg=PANEL_BG, fg=ACCENT,
                 font=("Consolas", 10, "bold")).pack(anchor="w")

        self.gauge_cpu = Gauge(sec2, "CPU", "%", maxval=100, warn=70, bad=90, accent="#00f2ff")
        self.gauge_cpu.pack(fill=tk.X, padx=10, pady=6)
        self.gauge_mem = Gauge(sec2, "MEMORY", "%", maxval=100, warn=75, bad=90, accent="#00ff88")
        self.gauge_mem.pack(fill=tk.X, padx=10, pady=6)
        self.gauge_disk = Gauge(sec2, "DISK", "%", maxval=100, warn=80, bad=95, accent="#bb86fc")
        self.gauge_disk.pack(fill=tk.X, padx=10, pady=6)
        self.gauge_proc = Gauge(sec2, "PROC MEM", "MB", maxval=500, warn=350, bad=450, accent="#ffaa00")
        self.gauge_proc.pack(fill=tk.X, padx=10, pady=6)

        self.card_lat = MetricCard(sec2, "INFER LATENCY", lambda p: p["latency_ms"], "{:.1f} ms", "#bb86fc")
        self.card_lat.pack(fill=tk.X, padx=10, pady=4)
        self.card_tick = MetricCard(sec2, "TICK", lambda p: p["step"], "{:d}", "#00f2ff")
        self.card_tick.pack(fill=tk.X, padx=10, pady=4)

        # Right column: graph + forecast + input list
        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        in_panel = tk.Frame(right, bg=PANEL_BG, height=100)
        in_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        in_panel.pack_propagate(False)
        tk.Label(in_panel, text="AVAILABLE INPUTS (input/ folder)", bg=PANEL_BG,
                 fg=ACCENT, font=("Consolas", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        self.in_listbox = tk.Listbox(in_panel, bg=PANEL_BG2, fg=TEXT, height=3,
                                     highlightthickness=0, selectbackground=ACCENT,
                                     selectforeground="#000", font=("Consolas", 9))
        self.in_listbox.pack(fill=tk.X, padx=10, pady=(0, 6))
        self._refresh_input_list()

        graph_frame = tk.Frame(right, bg=PANEL_BG)
        graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tk.Label(graph_frame, text="ACCURACY  —  ACTUAL vs PREDICTED", bg=PANEL_BG,
                 fg=ACCENT, font=("Consolas", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        self.fig = Figure(figsize=(7, 4.2), dpi=100, facecolor=PANEL_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL_BG2)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#222a3a")
        self.ax.set_xlabel("Steps", color=MUTED, fontsize=8)
        self.ax.set_ylabel("Demand (kW)", color=MUTED, fontsize=8)
        self.line_true, = self.ax.plot([], [], color="#00f2ff", label="Actual", lw=1.6, alpha=0.9)
        self.line_pred, = self.ax.plot([], [], color="#ff00ff", label="Predicted", lw=1.6, alpha=0.9, linestyle="--")
        self.ax.legend(facecolor=PANEL_BG2, edgecolor="#222a3a", labelcolor=TEXT, fontsize=8, loc="upper left")
        self.fig.tight_layout(pad=0.8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Forecast strip
        fc_frame = tk.Frame(right, bg=PANEL_BG, height=150)
        fc_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        fc_frame.pack_propagate(False)
        tk.Label(fc_frame, text="FORECAST HORIZON  (next steps, base model)", bg=PANEL_BG,
                 fg=ACCENT, font=("Consolas", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.fc_canvas = tk.Canvas(fc_frame, bg=PANEL_BG2, height=90, highlightthickness=0)
        self.fc_canvas.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.fc_label = tk.Label(fc_frame, text="Next prediction: -- kW", bg=PANEL_BG,
                                 fg=TEXT, font=("Consolas", 11, "bold"))
        self.fc_label.pack(anchor="w", padx=12, pady=(0, 8))

        # Current value display
        cur_frame = tk.Frame(right, bg=PANEL_BG2, height=44)
        cur_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        cur_frame.pack_propagate(False)
        self.lbl_name = tk.Label(cur_frame, text="DATA: --", bg=PANEL_BG2, fg=ACCENT,
                                 font=("Consolas", 10, "bold"))
        self.lbl_name.pack(side=tk.LEFT, padx=14, pady=10)
        self.lbl_true = tk.Label(cur_frame, text="Actual: -- kW", bg=PANEL_BG2, fg="#888",
                                 font=("Consolas", 11))
        self.lbl_true.pack(side=tk.LEFT, padx=14, pady=10)
        self.lbl_pred = tk.Label(cur_frame, text="Predicted: -- kW", bg=PANEL_BG2, fg=ACCENT,
                                 font=("Consolas", 11, "bold"))
        self.lbl_pred.pack(side=tk.LEFT, padx=14, pady=10)
        self.lbl_err = tk.Label(cur_frame, text="Error: -- kW", bg=PANEL_BG2, fg=WARN_COLOR,
                                font=("Consolas", 11))
        self.lbl_err.pack(side=tk.LEFT, padx=14, pady=10)
        self.lbl_rem = tk.Label(cur_frame, text="REMAINING: --", bg=PANEL_BG2, fg=MUTED,
                                font=("Consolas", 9))
        self.lbl_rem.pack(side=tk.RIGHT, padx=14, pady=10)

    def _refresh_input_list(self):
        self.in_listbox.delete(0, tk.END)
        for f in self.input_files:
            self.in_listbox.insert(tk.END, f)

    def _refresh_columns(self):
        csv_path = self._selected_csv()
        cols = detect_target_columns(csv_path) if csv_path else []
        self.column_combo.config(values=cols)
        if cols:
            self.column_combo.current(0)
            self.column_var.set(cols[0])
        else:
            self.column_var.set("")

    def _on_input_change(self, _e=None):
        self.input_files = list_input_files()
        labels = [display_name(f) for f in self.input_files] or ["(no inputs)"]
        self.input_combo.config(values=labels)
        self._refresh_input_list()
        self._refresh_columns()

    def _selected_csv(self):
        label = self.input_var.get()
        for f in self.input_files:
            if display_name(f) == label:
                return os.path.join(INPUT_DIR, f)
        return os.path.join(INPUT_DIR, self.input_files[0]) if self.input_files else None

    def _selected_column(self):
        return self.column_var.get() or None

    def start(self):
        if self.engine and self.engine.running.is_set():
            return
        csv_path = self._selected_csv()
        if not csv_path:
            messagebox.showerror("No input", "No input data file available in input/ folder.")
            return
        column = self._selected_column()
        if not column:
            messagebox.showerror("No column", "No target column selected.")
            return
        self.status_lbl.config(text="RUNNING", fg=OK_COLOR)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.input_combo.config(state=tk.DISABLED)
        self.column_combo.config(state=tk.DISABLED)
        self.lbl_name.config(text=f"DATA: {display_name(os.path.basename(csv_path))}  [{column}]")
        mem_prefix = os.path.join(
            INPUT_DIR,
            display_name(os.path.basename(csv_path)) + "_" + column + "_memory",
        )
        self.engine = SimEngine(csv_path, column, self.q,
                                speed_ms=int(self.speed_var.get()),
                                memory_path_prefix=mem_prefix)
        self.engine.start_sim()

    def pause(self):
        if not self.engine:
            return
        if self.engine.paused.is_set():
            self.engine.resume()
            self.btn_pause.config(text="⏸ PAUSE")
            self.status_lbl.config(text="RUNNING", fg=OK_COLOR)
        else:
            self.engine.pause()
            self.btn_pause.config(text="▶ RESUME")
            self.status_lbl.config(text="PAUSED", fg=WARN_COLOR)

    def stop(self):
        if self.engine:
            self.engine.stop()
            # wait for the engine thread to fully exit before saving memory
            self.engine.join(timeout=5)
            try:
                self.engine.predictor.save_memory(self.engine.memory_path_prefix)
            except Exception:
                pass
            self.engine = None
        self.status_lbl.config(text="IDLE", fg=MUTED)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ PAUSE")
        self.btn_stop.config(state=tk.DISABLED)
        self.input_combo.config(state="readonly")
        self.column_combo.config(state="readonly")

    def poll(self):
        try:
            while True:
                msg, payload = self.q.get_nowait()
                if msg == "tick":
                    self._on_tick(payload)
                elif msg == "error":
                    messagebox.showerror("Simulation Error", payload)
                    self.stop()
                elif msg == "done":
                    self.status_lbl.config(text="STREAM ENDED", fg=WARN_COLOR)
                    self.stop()
        except queue.Empty:
            pass
        self.poll_id = self.root.after(25, self.poll)

    def _on_tick(self, p):
        self.gauge_cpu.set(p["cpu"])
        self.gauge_mem.set(p["mem"])
        self.gauge_disk.set(p["disk"])
        self.gauge_proc.set(p["proc_mem_mb"])
        self.card_lat.update(p); self.card_lat.pulse()
        self.card_tick.update(p)

        self.lbl_true.config(text=f"Actual: {p['true']:.2f} kW")
        self.lbl_pred.config(text=f"Predicted: {p['pred']:.2f} kW")
        self.lbl_err.config(text=f"Error: {p['err']:.2f} kW",
                            fg=OK_COLOR if p['err'] < 0.5 else (WARN_COLOR if p['err'] < 2.0 else BAD_COLOR))
        self.lbl_rem.config(text=f"REMAINING: {p['remaining']}")

        # Graph
        xs = list(range(len(p["trues"])))
        self.line_true.set_data(xs, p["trues"])
        self.line_pred.set_data(xs, p["preds"])
        if xs:
            self.ax.set_xlim(0, max(len(xs) - 1, 1))
            ymin = min(min(p["trues"]), min(p["preds"]))
            ymax = max(max(p["trues"]), max(p["preds"]))
            pad = (ymax - ymin) * 0.15 + 0.01
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.fig.canvas.draw_idle()

        # Forecast strip
        self._draw_forecast(p["forecast"], p["pred"])
        self.fc_label.config(text=f"Next prediction: {p['pred']:.2f} kW")

    def _draw_forecast(self, forecast, current_pred):
        c = self.fc_canvas
        c.delete("fc_all")
        w = c.winfo_width() or 700
        h = c.winfo_height() or 90
        if not forecast:
            return
        arr = np.array(forecast, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            mx = mn + 1.0
        pts = []
        for i, v in enumerate(arr):
            x = (i / (len(arr) - 1)) * (w - 40) + 20 if len(arr) > 1 else w / 2
            y = h - 20 - ((v - mn) / (mx - mn + 1e-8)) * (h - 40)
            pts.append((x, y))
        for i in range(1, len(pts)):
            c.create_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1],
                          fill="#00ff88", width=2, tags="fc_all", capstyle=tk.ROUND)
        for x, y in pts:
            c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#00ff88", outline="", tags="fc_all")
        c.create_line(20, h - 20, pts[0][0], pts[0][1], fill=ACCENT, width=2, dash=(4, 4), tags="fc_all")
        c.create_text(10, h / 2, text=f"now: {current_pred:.1f}", fill=ACCENT,
                      font=("Consolas", 8), anchor="e", tags="fc_all")

    def run(self):
        self.root.mainloop()


def main():
    ensure_default_input()
    if not list_input_files():
        print("No input data found in input/ folder", file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    app = Dashboard(root)
    app.run()


if __name__ == "__main__":
    main()
