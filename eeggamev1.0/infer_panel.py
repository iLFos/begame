# -*- coding: utf-8 -*-
"""
infer_panel.py

eeggamev1.0 推理面板（白色简约风）。

布局：
  ┌─ 模型与游戏区 ───────────────────────────────────────────┐
  │  模型权重 (Browse)  [加载]                              │
  │  scaler (自动匹配)                                       │
  │  ▶ 启动游戏 / ■ 停止游戏                                │
  └──────────────────────────────────────────────────────────┘
  ┌─ 数据源区 (左 380px) ─┐  ┌─ 阈值与概率区 (中间自适应) ──┐
  │ [TGAM] COMx 57600 连接│  │ 阈值滑块 ─────●──── 50         │
  │ CSV: ... 浏览 回放   │  │ 当前概率柱状图                │
  │ 信号灯 (红/绿)       │  │ 当前状态：运动/静止             │
  └──────────────────────┘  └──────────────────────────────┘
  ┌─ 日志区 (底部 8 行 Text) ─────────────────────────────┐

阈值逻辑：
  thr = slider / 100
  仅当 prob[1] >= thr 时 movda.txt 写 "1"；否则 "0"。
"""
import os
import time
import queue
import threading
import subprocess
import sys
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from realtime_core import RealtimeInference, WINDOW_SIZE, CHANNELS
from tgam_serial import TGAMReader, list_ports as _list_ports


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "eeg_cnn_best_weights.npy")
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "be_stay_a_move_data",
                            "new_data", "move")
GAME_URL = "http://localhost:8080"

# 配色
BG = "#FFFFFF"
PANEL = "#F7F8FA"
PANEL_BORDER = "#E5E7EB"
ACCENT = "#2563EB"
ACCENT_DARK = "#1D4ED8"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
GRAY = "#9CA3AF"
TXT = "#1F1F1F"
TXT2 = "#6B7280"
TXT3 = "#9CA3AF"
ENTRY_BG = "#FFFFFF"
ENTRY_BORDER = "#D1D5DB"
BUTTON_BG = "#FFFFFF"

FONT_FAMILY = "Microsoft YaHei"
FONT_FALLBACK = "Segoe UI"

plt.rcParams["font.sans-serif"] = [FONT_FAMILY, FONT_FALLBACK, "Arial"]
plt.rcParams["axes.unicode_minus"] = False


class InferPanel(tk.Frame):
    """白色简约推理面板。"""

    def __init__(self, master, paths_dict=None, on_status=None,
                 on_paths_changed=None, **kw):
        super().__init__(master, bg=BG, **kw)
        self.on_status = on_status
        self.on_paths_changed = on_paths_changed

        # 全局共享路径（顶部数据目录 + 自定义配置都可读写）
        pd = paths_dict or {}
        self.train_data_var = tk.StringVar(value=pd.get("move_dir") or "")
        self.export_path_var = tk.StringVar(value=pd.get("export_dir") or "")

        # 状态
        self.inferencer = RealtimeInference(model_path=DEFAULT_WEIGHTS,
                                             threshold=0.5,
                                             on_result=self._on_result,
                                             on_log=self._on_log_safe)
        self.tgam = None
        self.csv_thread = None
        self.csv_stop = threading.Event()
        self.game_proc = None
        self.window_counter = 0

        # UI 变量
        self.threshold_var = tk.IntVar(value=50)
        self.weights_var = tk.StringVar(value=DEFAULT_WEIGHTS)
        self.csv_var = tk.StringVar(value="")
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="57600")
        self.signal_state = tk.StringVar(value="尚未开始")
        self.prob_stay = tk.DoubleVar(value=0.5)
        self.prob_move = tk.DoubleVar(value=0.5)
        self._port_list = []
        self._build_layout()
        self._refresh_ports()
        self._init_chart()

    def set_paths(self, paths_dict):
        """主窗口保存路径后回调：刷新自定义配置 StringVar。"""
        if not paths_dict:
            return
        if paths_dict.get("move_dir") is not None:
            self.train_data_var.set(paths_dict["move_dir"])
        if paths_dict.get("export_dir") is not None:
            self.export_path_var.set(paths_dict["export_dir"])

    def _save_custom_paths(self):
        """保存到 eeg_config.json。"""
        try:
            import config_store
            cur = config_store.load_config()
        except Exception:
            cur = {}
        cur["move_dir"] = self.train_data_var.get().strip()
        # stay_dir 在推理侧不强求；保留原值或清空
        if not cur.get("stay_dir"):
            cur["stay_dir"] = self.train_data_var.get().strip()
        cur["export_dir"] = self.export_path_var.get().strip()
        try:
            import config_store as _cs
            _cs.save_config(cur)
            self._set_status("已保存到 eeg_config.json", "ok")
            self.log_msg("已保存配置：导出=%s" % cur["export_dir"], "ok")
            # 通知主窗口刷新顶部 chip + 同步两个面板
            if self.on_paths_changed:
                self.on_paths_changed()
        except Exception as e:
            self._set_status("保存失败：%s" % e, "err")
            self.log_msg("保存失败：%s" % e, "err")

    # ====================== 工具 ====================== #
    def _set_status(self, text, level="info"):
        if self.on_status:
            self.on_status(text, level)

    def _on_log_safe(self, msg, level="info"):
        # realtime_core 可能在子线程调用 → after 到主线程
        self.after(0, lambda: self.log_msg(msg, level))

    # ====================== 布局 ====================== #
    def _build_layout(self):
        # 顶部：模型与游戏
        top = tk.Frame(self, bg=PANEL, bd=1, relief=tk.FLAT,
                       highlightthickness=1, highlightbackground=PANEL_BORDER)
        top.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(16, 8))
        self._build_top(top)

        # 底部日志（先 pack 到 BOTTOM，避免被中段 expand 区域挤出窗口）
        bot = tk.Frame(self, bg=PANEL, bd=1, relief=tk.FLAT,
                       highlightthickness=1, highlightbackground=PANEL_BORDER)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 16))
        self._build_log(bot)

        # 中段：左 数据源 + 右 阈值/概率
        mid = tk.Frame(self, bg=BG)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=8)
        self._build_left(mid)
        self._build_right(mid)

    def _build_top(self, parent):
        tk.Label(parent, text="模型与游戏", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor=tk.W,
                                                      padx=12, pady=(10, 4))

        # 模型权重
        row1 = tk.Frame(parent, bg=PANEL)
        row1.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(row1, text="模型权重", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9), width=10, anchor=tk.W).pack(side=tk.LEFT)
        ent = tk.Entry(row1, textvariable=self.weights_var, bg=ENTRY_BG,
                       fg=TXT, relief=tk.FLAT, bd=1, highlightthickness=1,
                       highlightbackground=ENTRY_BORDER,
                       highlightcolor=ACCENT,
                       font=(FONT_FAMILY, 9))
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        tk.Button(row1, text="浏览", bg=BUTTON_BG, fg=TXT,
                  activebackground=PANEL, relief=tk.FLAT, bd=1,
                  highlightthickness=1, highlightbackground=ENTRY_BORDER,
                  font=(FONT_FAMILY, 9), cursor="hand2",
                  command=self._browse_weights).pack(side=tk.LEFT,
                                                       padx=(6, 0), ipady=2)
        tk.Button(row1, text="加载", bg=ACCENT, fg="white",
                  activebackground=ACCENT_DARK, relief=tk.FLAT,
                  font=(FONT_FAMILY, 9, "bold"), cursor="hand2",
                  command=self._load_model).pack(side=tk.LEFT,
                                                  padx=(6, 0), ipady=2)

        # scaler（自动匹配，仅展示）
        row2 = tk.Frame(parent, bg=PANEL)
        row2.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(row2, text="scaler", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9), width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.scaler_lbl = tk.Label(row2, text="（未加载）", bg=PANEL, fg=TXT3,
                                    font=(FONT_FAMILY, 9), anchor=tk.W)
        self.scaler_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 游戏按钮
        row3 = tk.Frame(parent, bg=PANEL)
        row3.pack(fill=tk.X, padx=12, pady=(4, 12))
        self.game_start_btn = tk.Button(row3, text="▶ 启动游戏 (node server.js)",
                                         bg=ACCENT, fg="white",
                                         activebackground=ACCENT_DARK,
                                         relief=tk.FLAT,
                                         font=(FONT_FAMILY, 11, "bold"),
                                         cursor="hand2",
                                         command=self._start_game)
        self.game_start_btn.pack(side=tk.LEFT, ipadx=12, ipady=6)
        self.game_stop_btn = tk.Button(row3, text="■ 停止游戏", bg=BUTTON_BG,
                                        fg=TXT, activebackground=PANEL,
                                        relief=tk.FLAT,
                                        font=(FONT_FAMILY, 11, "bold"),
                                        cursor="hand2",
                                        command=self._stop_game,
                                        state=tk.DISABLED)
        self.game_stop_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=12, ipady=6)
        self.game_status = tk.Label(row3, text="游戏：未启动", bg=PANEL, fg=TXT3,
                                    font=(FONT_FAMILY, 9))
        self.game_status.pack(side=tk.LEFT, padx=(16, 0))

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=PANEL, bd=1, relief=tk.FLAT,
                        highlightthickness=1, highlightbackground=PANEL_BORDER,
                        width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        tk.Label(left, text="数据源", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor=tk.W,
                                                       padx=12, pady=(10, 4))

        # ---- TGAM ----
        tg = tk.LabelFrame(left, text="TGAM 串口", bg=PANEL, fg=TXT2,
                            font=(FONT_FAMILY, 10), bd=1,
                            relief=tk.FLAT)
        tg.pack(fill=tk.X, padx=12, pady=(4, 8))

        row = tk.Frame(tg, bg=PANEL)
        row.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(row, text="端口", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9)).pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(row, textvariable=self.port_var,
                                         values=self._port_list,
                                         state="readonly", width=10)
        self.port_combo.pack(side=tk.LEFT, padx=(4, 8))
        tk.Button(row, text="⟳", bg=BUTTON_BG, fg=TXT,
                  activebackground=PANEL, relief=tk.FLAT,
                  font=(FONT_FAMILY, 9), cursor="hand2",
                  command=self._refresh_ports).pack(side=tk.LEFT)
        tk.Label(row, text="波特率", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(12, 4))
        tk.Entry(row, textvariable=self.baud_var, width=8,
                 bg=ENTRY_BG, fg=TXT, relief=tk.FLAT, bd=1,
                 highlightthickness=1, highlightbackground=ENTRY_BORDER,
                 font=(FONT_FAMILY, 9)).pack(side=tk.LEFT)

        row = tk.Frame(tg, bg=PANEL)
        row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.tgam_btn = tk.Button(row, text="连接串口", bg=ACCENT, fg="white",
                                   activebackground=ACCENT_DARK, relief=tk.FLAT,
                                   font=(FONT_FAMILY, 10, "bold"),
                                   cursor="hand2",
                                   command=self._toggle_tgam)
        self.tgam_btn.pack(side=tk.LEFT, ipadx=10, ipady=3)
        tk.Label(row, textvariable=self.port_var, bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(8, 0))

        # ---- CSV ----
        cg = tk.LabelFrame(left, text="离线 CSV 回放", bg=PANEL, fg=TXT2,
                            font=(FONT_FAMILY, 10), bd=1, relief=tk.FLAT)
        cg.pack(fill=tk.X, padx=12, pady=(0, 8))
        row = tk.Frame(cg, bg=PANEL)
        row.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(row, text="CSV 文件", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9)).pack(anchor=tk.W)
        er = tk.Frame(cg, bg=PANEL)
        er.pack(fill=tk.X, padx=8, pady=(0, 6))
        ent = tk.Entry(er, textvariable=self.csv_var, bg=ENTRY_BG, fg=TXT,
                       relief=tk.FLAT, bd=1, highlightthickness=1,
                       highlightbackground=ENTRY_BORDER,
                       highlightcolor=ACCENT,
                       font=(FONT_FAMILY, 9))
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        tk.Button(er, text="浏览", bg=BUTTON_BG, fg=TXT,
                  activebackground=PANEL, relief=tk.FLAT, bd=1,
                  highlightthickness=1, highlightbackground=ENTRY_BORDER,
                  font=(FONT_FAMILY, 9), cursor="hand2",
                  command=self._browse_csv).pack(side=tk.LEFT, padx=(6, 0))
        self.csv_btn = tk.Button(cg, text="▶ 开始回放", bg=ACCENT, fg="white",
                                  activebackground=ACCENT_DARK, relief=tk.FLAT,
                                  font=(FONT_FAMILY, 10, "bold"),
                                  cursor="hand2", command=self._toggle_csv)
        self.csv_btn.pack(anchor=tk.W, padx=8, pady=(0, 8), ipadx=8, ipady=3)

        # ---- 信号灯 ----
        lamp_frame = tk.Frame(left, bg=PANEL)
        lamp_frame.pack(fill=tk.X, padx=12, pady=(4, 8))
        tk.Label(lamp_frame, text="状态指示", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        self.lamp_canvas = tk.Canvas(lamp_frame, width=28, height=28,
                                      bg=PANEL, highlightthickness=0)
        self.lamp_canvas.pack(side=tk.LEFT, padx=(10, 6))
        self._lamp_id = self.lamp_canvas.create_oval(4, 4, 24, 24,
                                                     fill=GRAY, outline="")
        self.signal_lbl = tk.Label(lamp_frame, textvariable=self.signal_state,
                                    bg=PANEL, fg=TXT, font=(FONT_FAMILY, 10, "bold"))
        self.signal_lbl.pack(side=tk.LEFT)

        # 快捷入口：使用训练数据默认路径
        tk.Label(left, text="说明：",
                 bg=PANEL, fg=TXT2, font=(FONT_FAMILY, 9)).pack(anchor=tk.W,
                                                                padx=12,
                                                                pady=(8, 2))
        info = ("• 第一次使用请先在「AI 训练程序」导出模型。\n"
                "• 离线 CSV 适用于无硬件回放验证；TGAM 需 pyserial。\n"
                "• 阈值滑块控制运动概率最小值。")
        tk.Label(left, text=info, bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 9), justify=tk.LEFT,
                 wraplength=360).pack(anchor=tk.W, padx=12, pady=(0, 8))

        # ---- 自定义配置（持久化路径）----
        self._build_custom_config(left)

    def _build_custom_config(self, parent):
        """自定义配置：训练数据路径 + 模型导出位置，写入 eeg_config.json。"""
        from tkinter import filedialog as _fd
        box = tk.LabelFrame(parent, text="  自定义配置  ",
                            bg=PANEL, fg=ACCENT, bd=1, relief=tk.FLAT,
                            font=(FONT_FAMILY, 10, "bold"),
                            highlightthickness=1,
                            highlightbackground=PANEL_BORDER)
        box.pack(fill=tk.X, padx=12, pady=(8, 8), ipady=2)

        def add_path_row(label, var):
            r = tk.Frame(box, bg=PANEL)
            r.pack(fill=tk.X, padx=8, pady=(4, 2))
            tk.Label(r, text=label, bg=PANEL, fg=TXT2,
                     font=(FONT_FAMILY, 9)).pack(anchor=tk.W)
            sub = tk.Frame(r, bg=PANEL)
            sub.pack(fill=tk.X, pady=(2, 0))
            ent = tk.Entry(sub, textvariable=var, bg=ENTRY_BG, fg=TXT,
                           relief=tk.FLAT, bd=1, highlightthickness=1,
                           highlightbackground=ENTRY_BORDER,
                           highlightcolor=ACCENT,
                           font=(FONT_FAMILY, 9))
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
            btn = tk.Button(sub, text="浏览", bg=BUTTON_BG, fg=TXT,
                            activebackground=PANEL, relief=tk.FLAT, bd=1,
                            highlightthickness=1,
                            highlightbackground=ENTRY_BORDER,
                            font=(FONT_FAMILY, 9), cursor="hand2",
                            command=lambda: _browse(var))
            btn.pack(side=tk.RIGHT, padx=(6, 0), ipady=1)

        def _browse(var):
            d = _fd.askdirectory(initialdir=var.get() or BASE_DIR,
                                  title="选择目录")
            if d:
                var.set(d)

        add_path_row("训练数据路径（move）", self.train_data_var)
        add_path_row("模型导出位置", self.export_path_var)

        sep = tk.Frame(box, bg=PANEL_BORDER, height=1)
        sep.pack(fill=tk.X, padx=8, pady=(6, 4))

        save_btn = tk.Button(box, text="💾 保存到全局配置", bg=ACCENT, fg="white",
                             activebackground=ACCENT_DARK,
                             activeforeground="white",
                             relief=tk.FLAT, font=(FONT_FAMILY, 10, "bold"),
                             cursor="hand2", command=self._save_custom_paths)
        save_btn.pack(fill=tk.X, padx=8, pady=(0, 8), ipady=4)

        tk.Label(box, text="保存到 eeg_config.json，下次启动自动加载",
                 bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 8)).pack(anchor=tk.W, padx=8, pady=(0, 4))

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=PANEL, bd=1, relief=tk.FLAT,
                         highlightthickness=1, highlightbackground=PANEL_BORDER)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 阈值区
        thr_frame = tk.Frame(right, bg=PANEL)
        thr_frame.pack(fill=tk.X, padx=16, pady=(12, 6))
        tk.Label(thr_frame, text="运动概率阈值", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(side=tk.LEFT)

        self.thr_value_lbl = tk.Label(thr_frame, text="0.50", bg=PANEL, fg=ACCENT,
                                       font=(FONT_FAMILY, 14, "bold"))
        self.thr_value_lbl.pack(side=tk.RIGHT)

        # 滑块 + Spinbox 双向绑定
        slider_row = tk.Frame(right, bg=PANEL)
        slider_row.pack(fill=tk.X, padx=16, pady=(0, 8))
        self.thr_scale = tk.Scale(slider_row, from_=1, to=100, orient=tk.HORIZONTAL,
                                    variable=self.threshold_var,
                                    bg=PANEL, fg=TXT, highlightthickness=0,
                                    troughcolor="#E5E7EB", activebackground=ACCENT,
                                    showvalue=False, length=380,
                                    command=lambda v: self._sync_threshold("scale"))
        self.thr_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.thr_spin = tk.Spinbox(slider_row, from_=1, to=100, increment=1,
                                    textvariable=self.threshold_var, width=5,
                                    bg=ENTRY_BG, fg=TXT, relief=tk.FLAT, bd=1,
                                    highlightthickness=1,
                                    highlightbackground=ENTRY_BORDER,
                                    highlightcolor=ACCENT,
                                    font=(FONT_FAMILY, 12, "bold"),
                                    command=lambda: self._sync_threshold("spin"))
        self.thr_spin.pack(side=tk.LEFT, padx=(8, 0))
        self.threshold_var.trace_add("write", lambda *_: self._sync_threshold("var"))

        # 概率柱状图
        self.fig = Figure(figsize=(5, 3.0), dpi=100, facecolor=PANEL)
        self.ax_prob = self.fig.add_subplot(111, facecolor="#FFFFFF")
        self.canvas_prob = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_prob.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                              padx=16, pady=(0, 10))
        self._style_prob_axes()
        self.fig.tight_layout(pad=2.0)

        # 当前状态
        state_row = tk.Frame(right, bg=PANEL)
        state_row.pack(fill=tk.X, padx=16, pady=(0, 8))
        tk.Label(state_row, text="当前窗口", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9)).pack(side=tk.LEFT)
        self.window_lbl = tk.Label(state_row, text="0", bg=PANEL, fg=TXT,
                                    font=(FONT_FAMILY, 10, "bold"))
        self.window_lbl.pack(side=tk.LEFT, padx=(6, 0))

    def _build_log(self, parent):
        tk.Label(parent, text="日志", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor=tk.W,
                                                       padx=12, pady=(8, 2))
        self.log = tk.Text(parent, bg="#FFFFFF", fg=TXT, bd=0,
                           highlightthickness=1,
                           highlightbackground=PANEL_BORDER,
                           font=("Consolas", 9), wrap=tk.WORD,
                           state=tk.DISABLED, height=8)
        self.log.pack(fill=tk.X, padx=12, pady=(0, 10))

    # ====================== 样式 ====================== #
    def _style_prob_axes(self):
        ax = self.ax_prob
        ax.clear()
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["静止", "运动"], color=TXT, fontname=FONT_FAMILY)
        ax.tick_params(colors=TXT2)
        for sp in ax.spines.values():
            sp.set_color(PANEL_BORDER)
        ax.grid(True, axis="y", color="#F0F0F0", linewidth=0.6)
        ax.set_title("当前窗口概率分布", color=TXT, fontname=FONT_FAMILY, fontsize=11)
        self.bars = ax.bar([0, 1], [0.5, 0.5],
                            color=[GREEN, RED], alpha=0.85,
                            edgecolor="#FFFFFF", linewidth=1)
        self.canvas_prob.draw()

    def _init_chart(self):
        self._style_prob_axes()

    # ====================== 模型加载 ====================== #
    def _browse_weights(self):
        f = filedialog.askopenfilename(initialdir=BASE_DIR,
                                        title="选择模型权重 (.npy)",
                                        filetypes=[("NumPy weights", "*.npy"),
                                                    ("All", "*.*")])
        if f:
            self.weights_var.set(f)

    def _load_model(self):
        try:
            self.inferencer.model_path = self.weights_var.get().strip()
            self.inferencer.load_model()
            self.scaler_lbl.config(
                text="已加载：%s   |   %s" % (
                    os.path.basename(self.inferencer.model_path),
                    os.path.basename(self._find_scaler_path(
                        self.inferencer.model_path, "scaler_mean.npy"))))
            self.log_msg("模型已加载", "ok")
            self._set_status("模型已加载", "ok")
        except Exception as e:
            messagebox.showerror("加载失败", str(e))
            self.log_msg("加载失败：%s" % e, "err")

    @staticmethod
    def _find_scaler_path(model_path, name):
        base = os.path.splitext(model_path)[0]
        cand = [base + "_" + name, os.path.join(BASE_DIR, name)]
        for p in cand:
            if os.path.exists(p):
                return p
        return cand[-1]

    # ====================== TGAM ====================== #
    def _refresh_ports(self):
        self._port_list = _list_ports() or []
        if hasattr(self, "port_combo"):
            self.port_combo["values"] = self._port_list
            if self._port_list and not self.port_var.get():
                self.port_var.set(self._port_list[0])

    def _toggle_tgam(self):
        if self.tgam and self.tgam.is_open():
            self.tgam.close()
            self.tgam = None
            self.tgam_btn.config(text="连接串口", bg=ACCENT)
            self.log_msg("TGAM 已断开", "info")
            self._set_status("TGAM 已断开", "info")
            return
        if not self._port_list:
            messagebox.showwarning("无可用串口", "未检测到任何串口，请连接 TGAM 后重试。")
            return
        port = self.port_var.get()
        try:
            baud = int(self.baud_var.get())
        except Exception:
            baud = 57600
        try:
            self.tgam = TGAMReader(port, baud=baud,
                                     on_packet=self._on_tgam_packet,
                                     on_status=self._on_log_safe)
            self.tgam.open()
            self.tgam_btn.config(text="■ 断开串口", bg=BUTTON_BG, fg=TXT)
            self._set_status("TGAM 已连接 %s" % port, "ok")
        except Exception as e:
            messagebox.showerror("串口错误", str(e))
            self.log_msg("串口错误：%s" % e, "err")

    def _on_tgam_packet(self, packet):
        vec = TGAMReader.build_feature_vector(packet)
        self.inferencer.add_feature_vector(vec)

    # ====================== CSV ====================== #
    def _browse_csv(self):
        f = filedialog.askopenfilename(initialdir=DEFAULT_CSV,
                                        title="选择 CSV",
                                        filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if f:
            self.csv_var.set(f)

    def _toggle_csv(self):
        if self.csv_thread and self.csv_thread.is_alive():
            self.csv_stop.set()
            self.csv_btn.config(text="▶ 开始回放", bg=ACCENT)
            self.log_msg("CSV 回放已停止", "info")
            return
        path = self.csv_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("路径错误", "请先选择有效的 CSV 文件。")
            return
        self.csv_stop.clear()
        self.csv_btn.config(text="■ 停止回放", bg=BUTTON_BG, fg=TXT)
        self.window_counter = 0
        self.csv_thread = threading.Thread(target=self._csv_replay_worker,
                                            args=(path,), daemon=True)
        self.csv_thread.start()
        self.log_msg("CSV 回放开始：%s" % os.path.basename(path), "info")
        self._set_status("CSV 回放中", "ok")

    def _csv_replay_worker(self, path):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0)
            if sig.shape[1] > CHANNELS:
                sig = sig[:, :CHANNELS]
            elif sig.shape[1] < CHANNELS:
                pad = np.zeros((sig.shape[0], CHANNELS - sig.shape[1]),
                                dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=1)
            n = sig.shape[0]
            stride = 4
            for s in range(0, n - WINDOW_SIZE + 1, stride):
                if self.csv_stop.is_set():
                    break
                w = sig[s:s + WINDOW_SIZE]
                self.inferencer.infer_window(w)
                time.sleep(0.06)
        except Exception as e:
            self.after(0, lambda: self.log_msg("CSV 回放出错：%s" % e, "err"))
        finally:
            self.after(0, lambda: self.csv_btn.config(
                text="▶ 开始回放", bg=ACCENT))

    # ====================== 阈值 ====================== #
    def _sync_threshold(self, src):
        try:
            v = int(float(self.threshold_var.get()))
            v = max(1, min(100, v))
        except Exception:
            v = 50
        if src != "var":
            self.threshold_var.set(v)
        thr = v / 100.0
        self.thr_value_lbl.config(text="%.2f" % thr)
        self.inferencer.set_threshold(thr)
        if src == "var":
            self.log_msg("阈值更新：%d/100 = %.2f" % (v, thr), "info")

    # ====================== 游戏控制 ====================== #
    def _start_game(self):
        if self.game_proc is not None:
            return
        try:
            self.game_proc = subprocess.Popen(
                ["node", "server.js"],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                                if sys.platform == "win32" else 0,
            )
        except FileNotFoundError:
            messagebox.showerror(
                "未找到 node",
                "未检测到 node.exe。请先安装 Node.js 并加入 PATH，"
                "或使用 start.bat 手动启动服务器。")
            self.game_proc = None
            return
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self.game_proc = None
            return

        self.game_start_btn.config(state=tk.DISABLED)
        self.game_stop_btn.config(state=tk.NORMAL)
        self.game_status.config(text="游戏：已启动（端口 8080）", fg=GREEN)
        self.log_msg("游戏服务器已启动：%s" % GAME_URL, "ok")
        self._set_status("游戏已启动", "ok")

        # 1.5s 后自动打开浏览器
        def _open_browser():
            try:
                webbrowser.open(GAME_URL)
            except Exception:
                pass
        self.after(1500, _open_browser)

    def _stop_game(self):
        if self.game_proc is None:
            return
        try:
            self.game_proc.terminate()
            try:
                self.game_proc.wait(timeout=2)
            except Exception:
                self.game_proc.kill()
        except Exception as e:
            self.log_msg("停止游戏失败：%s" % e, "warn")
        finally:
            self.game_proc = None
            self.game_start_btn.config(state=tk.NORMAL)
            self.game_stop_btn.config(state=tk.DISABLED)
            self.game_status.config(text="游戏：未启动", fg=TXT3)
            self.log_msg("游戏服务器已停止", "info")
            self._set_status("游戏已停止", "info")

    # ====================== 推理结果回调 ====================== #
    def _on_result(self, pred, prob_move, probs, win12):
        # 在主线程里更新 UI
        self.after(0, lambda: self._update_ui(pred, prob_move, probs))

    def _update_ui(self, pred, prob_move, probs):
        self.window_counter += 1
        self.window_lbl.config(text=str(self.window_counter))
        self.prob_stay.set(probs[0])
        self.prob_move.set(probs[1])
        # 柱状图
        self.bars[0].set_height(probs[0])
        self.bars[1].set_height(probs[1])
        self.canvas_prob.draw_idle()
        # 信号灯 + 状态文字
        thr = self.threshold_var.get() / 100.0
        is_motion = (prob_move >= thr)
        color = RED if is_motion else GREEN
        self.lamp_canvas.itemconfig(self._lamp_id, fill=color)
        if is_motion:
            self.signal_state.set("运动（红方前进）")
            self.signal_lbl.config(fg=RED)
        else:
            self.signal_state.set("静止（红方停止）")
            self.signal_lbl.config(fg=TXT2)

    # ====================== 日志 ====================== #
    def log_msg(self, text, level="info"):
        color = {"info": TXT2, "ok": GREEN, "warn": AMBER,
                 "err": RED}.get(level, TXT2)
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n", level)
        self.log.tag_config(level, foreground=color)
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    # ====================== 退出时清理 ====================== #
    def cleanup(self):
        if self.tgam:
            try:
                self.tgam.close()
            except Exception:
                pass
        if self.csv_stop:
            self.csv_stop.set()
        self._stop_game()


def main():
    root = tk.Tk()
    root.title("eeggamev1.0 · AI 推理检测程序")
    root.geometry("1180x800")
    root.configure(bg=BG)
    panel = InferPanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()