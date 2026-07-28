# -*- coding: utf-8 -*-
"""
train_panel.py

eeggamev1.0 训练面板（白色简约风）。

约束（v1.0）：
  - 用户只暴露「卷积核数 f1」一个控件，f2 自动 = 2 × f1。
  - 其它可调超参：epochs / lr / batch / window / stride / k1 / k2。
  - move / stay 目录可自定义；导出目录可自定义。
  - 子线程训练 + queue 刷新 loss / acc 曲线，停止事件支持优雅中断。

风格：
  主背景 #FFFFFF，面板次背景 #F7F8FA，分隔线 #E5E7EB，主文字 #1F1F1F，
  次文字 #6B7280，占位 #9CA3AF，强调蓝 #2563EB，成功绿 #10B981，
  警告橙 #F59E0B，错误红 #EF4444。
"""
import os
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_utils import (load_dataset, dataset_summary, csv_summary,
                        export_dataset_csv, CHANNELS,
                        DEFAULT_WINDOW, DEFAULT_STRIDE)
from model import EEGCNN
from model_ops import Scaler

# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_MOVE_DIR = os.path.join(PROJECT_ROOT, "be_stay_a_move_data", "new_data", "move")
DEFAULT_STAY_DIR = os.path.join(PROJECT_ROOT, "be_stay_a_move_data", "new_data", "stay")
DEFAULT_EXPORT_DIR = BASE_DIR

# 配色（白色简约风）
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
BUTTON_BORDER = "#D1D5DB"

FONT_FAMILY = "Microsoft YaHei"
FONT_FALLBACK = "Segoe UI"

plt.rcParams["font.sans-serif"] = [FONT_FAMILY, FONT_FALLBACK, "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 默认超参数（与原 EEG_Football/train_model.py 一致）
DEFAULTS = {
    "f1": 4, "k1": 5, "k2": 3,
    "lr": 0.3, "epochs": 500, "batch": 0,
    "window": DEFAULT_WINDOW, "stride": DEFAULT_STRIDE,
}

REDRAW_EVERY = 5


class TrainPanel(tk.Frame):
    """白色简约训练面板，可作为顶层 Frame 嵌入主窗口。"""

    def __init__(self, master, paths_dict=None, on_status=None, **kw):
        super().__init__(master, bg=BG, **kw)
        self.on_status = on_status  # 回调(text, level) -> 主窗口状态栏

        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.training = False
        self.model = None
        self.scaler = None

        self.epochs_log = []
        self.train_start_ts = 0.0
        self.last_epoch_ts = 0.0

        # 数据目录 StringVar（全局共享，由 main.py 注入并可热更新）
        pd = paths_dict or {}
        self.move_var = tk.StringVar(value=pd.get("move_dir") or DEFAULT_MOVE_DIR)
        self.stay_var = tk.StringVar(value=pd.get("stay_dir") or DEFAULT_STAY_DIR)
        self.export_var = tk.StringVar(value=pd.get("export_dir") or DEFAULT_EXPORT_DIR)

        self._build_layout()
        self._refresh_dataset_info()
        self._poll_queue()

    def set_paths(self, paths_dict):
        """主窗口保存路径后回调：刷新 3 个 StringVar + 重新统计。"""
        if not paths_dict:
            return
        if paths_dict.get("move_dir"):
            self.move_var.set(paths_dict["move_dir"])
        if paths_dict.get("stay_dir"):
            self.stay_var.set(paths_dict["stay_dir"])
        if paths_dict.get("export_dir"):
            self.export_var.set(paths_dict["export_dir"])
        try:
            self._refresh_dataset_info()
        except Exception:
            pass

    # ====================== 工具方法 ====================== #
    def _set_status(self, text, level="info"):
        if self.on_status:
            self.on_status(text, level)

    def _fmt_time(self, sec):
        if sec <= 0:
            return "--"
        sec = int(sec)
        if sec < 60:
            return "%ds" % sec
        m, s = divmod(sec, 60)
        if m < 60:
            return "%dm%02ds" % (m, s)
        h, m = divmod(m, 60)
        return "%dh%02dm" % (h, m)

    # ====================== 布局 ====================== #
    def _build_layout(self):
        # 左侧：参数 + 目录 + 控制（固定宽度侧栏）
        # 右侧：大尺寸训练曲线 + 底部日志条
        side = tk.Frame(self, bg=PANEL, bd=1, relief=tk.FLAT,
                        highlightthickness=1, highlightbackground=PANEL_BORDER,
                        width=380)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 8), pady=16)
        side.pack_propagate(False)
        self._build_sidebar(side)

        right = tk.Frame(self, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                   padx=(0, 16), pady=16)
        self._build_plot_area(right)

    def _build_sidebar(self, parent):
        # ---- 控制按钮 + 进度（先 pack 到底部，永不被挤出）----
        ctrl = tk.Frame(parent, bg=PANEL)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(4, 12))

        self.prog_var = tk.DoubleVar(value=0)
        self.prog = ttk.Progressbar(ctrl, variable=self.prog_var, maximum=100,
                                    style="EEG.Horizontal.TProgressbar")
        self.prog.pack(fill=tk.X, pady=(0, 2))
        self.prog_lbl = tk.Label(ctrl, text="0 / 0", bg=PANEL, fg=TXT2,
                                 font=(FONT_FAMILY, 9))
        self.prog_lbl.pack(anchor=tk.E, pady=(0, 6))

        btns = tk.Frame(ctrl, bg=PANEL)
        btns.pack(fill=tk.X)
        self.start_btn = tk.Button(btns, text="▶  开始训练", bg=ACCENT, fg="white",
                                   activebackground=ACCENT_DARK,
                                   activeforeground="white",
                                   relief=tk.FLAT, font=(FONT_FAMILY, 11, "bold"),
                                   cursor="hand2", command=self.on_start)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7)

        self.stop_btn = tk.Button(btns, text="■  停止", bg=BUTTON_BG, fg=TXT,
                                  activebackground=PANEL,
                                  relief=tk.FLAT, font=(FONT_FAMILY, 11, "bold"),
                                  cursor="hand2", command=self.on_stop,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True,
                           ipady=7, padx=(8, 0))

        # ---- 参数区 ----
        tk.Label(parent, text="训练参数", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor=tk.W,
                                                      padx=14, pady=(12, 6))
        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(fill=tk.X, padx=14)

        row1 = tk.Frame(grid, bg=PANEL); row1.pack(fill=tk.X, pady=(0, 6))
        self._make_field(row1, "轮次 epochs", "epochs", DEFAULTS["epochs"],
                         width=8)
        self._make_field(row1, "学习率 lr", "lr", DEFAULTS["lr"],
                         width=8, is_float=True)
        self._make_field(row1, "卷积核 f1", "f1", DEFAULTS["f1"], width=6)

        row2 = tk.Frame(grid, bg=PANEL); row2.pack(fill=tk.X, pady=(0, 6))
        f2_box = tk.Frame(row2, bg=PANEL)
        f2_box.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(f2_box, text="卷积核 f2 (=2×f1)", bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9)).pack(anchor=tk.W)
        self.f2_var = tk.StringVar(value=str(DEFAULTS["f1"] * 2))
        tk.Label(f2_box, textvariable=self.f2_var, bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor=tk.W, pady=(4, 0))
        self.params["f1"].trace_add("write", lambda *_: self._sync_f2())
        self._make_field(row2, "核尺寸 k1", "k1", DEFAULTS["k1"], width=6)
        self._make_field(row2, "核尺寸 k2", "k2", DEFAULTS["k2"], width=6)

        row3 = tk.Frame(grid, bg=PANEL); row3.pack(fill=tk.X, pady=(0, 6))
        self._make_field(row3, "批大小", "batch", DEFAULTS["batch"], width=6)
        self._make_field(row3, "窗口", "window", DEFAULTS["window"], width=6)
        self._make_field(row3, "步长", "stride", DEFAULTS["stride"], width=6)

        tk.Frame(parent, bg=PANEL_BORDER, height=1).pack(fill=tk.X,
                                                          padx=14, pady=8)

        # ---- 信号预处理（过滤杂波）----
        self._build_preprocess_group(parent)

        tk.Frame(parent, bg=PANEL_BORDER, height=1).pack(fill=tk.X,
                                                          padx=14, pady=8)

        # ---- CSV 工具 ----
        self._build_csv_tools(parent)

        # ---- 数据集统计 ----
        head = tk.Frame(parent, bg=PANEL)
        head.pack(fill=tk.X, padx=14, pady=(4, 2))
        tk.Label(head, text="数据集统计", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 10, "bold")).pack(side=tk.LEFT)
        refresh = tk.Button(head, text="⟳ 重新统计", bg=ACCENT, fg="white",
                            activebackground=ACCENT_DARK,
                            activeforeground="white",
                            relief=tk.FLAT, font=(FONT_FAMILY, 8),
                            cursor="hand2", command=self._refresh_dataset_info)
        refresh.pack(side=tk.RIGHT, ipadx=6, ipady=1)

        self.info_var = tk.StringVar(value="数据集：加载中…")
        tk.Label(parent, textvariable=self.info_var, bg=PANEL, fg=AMBER,
                 font=(FONT_FAMILY, 9), justify=tk.LEFT, wraplength=340,
                 anchor=tk.NW).pack(fill=tk.X, padx=14, pady=(0, 4))

        # ---- 本次配置确认 ----
        self.conf_var = tk.StringVar(value="（点开始训练后显示已生效配置）")
        tk.Label(parent, textvariable=self.conf_var, bg=ENTRY_BG, fg=TXT,
                 font=(FONT_FAMILY, 9), anchor=tk.NW, justify=tk.LEFT,
                 wraplength=330).pack(fill=tk.X, padx=14, pady=(0, 8),
                                      ipadx=6, ipady=6)

    def _build_preprocess_group(self, parent):
        """信号预处理（过滤杂波）分组。默认全部关闭。"""
        box = tk.LabelFrame(parent, text="  信号预处理（过滤杂波）  ",
                            bg=PANEL, fg=ACCENT, bd=1, relief=tk.FLAT,
                            font=(FONT_FAMILY, 10, "bold"),
                            highlightthickness=1,
                            highlightbackground=PANEL_BORDER)
        box.pack(fill=tk.X, padx=14, pady=(0, 4), ipady=2)

        tk.Label(box, text="默认全关 → 与原训练结果一致；signalQuality 通道跳过滤波",
                 bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 8)).pack(anchor=tk.W, padx=8, pady=(2, 4))

        self.preproc = {
            "detrend": tk.BooleanVar(value=False),
            "bandpass": tk.BooleanVar(value=False),
            "notch": tk.BooleanVar(value=False),
            "clip": tk.BooleanVar(value=False),
            "fs": tk.StringVar(value="128"),
            "low": tk.StringVar(value="1.0"),
            "high": tk.StringVar(value="40.0"),
            "notch_freq": tk.StringVar(value="50"),
            "notch_q": tk.StringVar(value="30"),
            "clip_sigma": tk.StringVar(value="3.5"),
        }
        self._preproc_spins = []

        def make_check(parent, text, var, row, col):
            return tk.Checkbutton(parent, text=text, variable=var,
                                  bg=PANEL, fg=TXT, activebackground=PANEL,
                                  selectcolor=ENTRY_BG,
                                  font=(FONT_FAMILY, 9), anchor=tk.W)

        def make_param(parent, label, var, row, col, width=7):
            sub = tk.Frame(parent, bg=PANEL)
            sub.grid(row=row, column=col, sticky=tk.W, padx=4, pady=1)
            tk.Label(sub, text=label, bg=PANEL, fg=TXT2,
                     font=(FONT_FAMILY, 8)).pack(side=tk.LEFT)
            sp = tk.Spinbox(sub, textvariable=var, from_=0.0, to=9999.0,
                            increment=0.5, width=width, bg=ENTRY_BG, fg=TXT,
                            relief=tk.FLAT, bd=1, highlightthickness=1,
                            highlightbackground=ENTRY_BORDER,
                            highlightcolor=ACCENT,
                            buttonbackground=PANEL,
                            font=(FONT_FAMILY, 9), justify=tk.CENTER)
            sp.pack(side=tk.LEFT, padx=(2, 0), ipady=1)
            self._preproc_spins.append(sp)
            return sp

        g = tk.Frame(box, bg=PANEL); g.pack(fill=tk.X, padx=4, pady=(0, 4))
        for i in range(4):
            g.grid_columnconfigure(i, weight=1)

        c1 = make_check(g, "去基线 detrend", self.preproc["detrend"], 0, 0)
        c1.grid(row=0, column=0, sticky=tk.W, padx=6, pady=1)
        c2 = make_check(g, "带通 bandpass", self.preproc["bandpass"], 0, 2)
        c2.grid(row=0, column=2, sticky=tk.W, padx=6, pady=1)
        make_param(g, "低 Hz", self.preproc["low"], 1, 0)
        make_param(g, "高 Hz", self.preproc["high"], 1, 1)
        make_param(g, "fs Hz", self.preproc["fs"], 1, 2)
        make_param(g, "Q", self.preproc["notch_q"], 1, 3, width=5)

        c3 = make_check(g, "工频陷波 notch", self.preproc["notch"], 2, 0)
        c3.grid(row=2, column=0, sticky=tk.W, padx=6, pady=1)
        make_param(g, "陷波 Hz", self.preproc["notch_freq"], 3, 0, width=6)
        c4 = make_check(g, "异常值裁剪 clip", self.preproc["clip"], 2, 2)
        c4.grid(row=2, column=2, sticky=tk.W, padx=6, pady=1)
        make_param(g, "σ 倍数", self.preproc["clip_sigma"], 3, 2, width=6)

        # 把所有 spinbox 放进 _widget_refs 以便 _set_controls_enabled 控灰
        for k in ("fs", "low", "high", "notch_freq", "notch_q", "clip_sigma"):
            self._widget_refs.append(self.preproc[k])

    def _build_csv_tools(self, parent):
        """CSV 数据工具：浏览预览 + 导出数据集。"""
        box = tk.LabelFrame(parent, text="  CSV 数据工具  ",
                            bg=PANEL, fg=ACCENT, bd=1, relief=tk.FLAT,
                            font=(FONT_FAMILY, 10, "bold"),
                            highlightthickness=1,
                            highlightbackground=PANEL_BORDER)
        box.pack(fill=tk.X, padx=14, pady=(6, 4), ipady=2)

        row = tk.Frame(box, bg=PANEL); row.pack(fill=tk.X, padx=6, pady=4)
        b1 = tk.Button(row, text="📂 浏览并预览 CSV", bg=ACCENT, fg="white",
                       activebackground=ACCENT_DARK,
                       activeforeground="white",
                       relief=tk.FLAT, font=(FONT_FAMILY, 9, "bold"),
                       cursor="hand2", command=self._preview_csv)
        b1.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        b2 = tk.Button(row, text="⤓ 导出数据集", bg=BUTTON_BG, fg=TXT,
                       activebackground=PANEL, relief=tk.FLAT,
                       bd=1, highlightthickness=1,
                       highlightbackground=ENTRY_BORDER,
                       font=(FONT_FAMILY, 9, "bold"),
                       cursor="hand2", command=self._export_dataset)
        b2.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(6, 0))
        self._widget_refs += [b1, b2]

        tk.Label(box, text="预览：显示通道统计与前 2 秒波形（不修改训练数据）",
                 bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 8)).pack(anchor=tk.W, padx=8, pady=(0, 2))
        tk.Label(box, text="导出：把 move+stay 合并为单个 CSV（带 label）",
                 bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 8)).pack(anchor=tk.W, padx=8, pady=(0, 2))

    def _make_field(self, parent, label, key, default, side=tk.LEFT,
                    width=8, is_float=False):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(side=side, padx=(0, 12))
        tk.Label(f, text=label, bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 9)).pack(anchor=tk.W)
        var = tk.StringVar(value=str(default))
        if not hasattr(self, "params"):
            self.params = {}
        self.params[key] = var
        spin = tk.Spinbox(
            f, textvariable=var, from_=1, to=9999, increment=1, width=width,
            bg=ENTRY_BG, fg=TXT, relief=tk.FLAT, bd=1,
            highlightthickness=1, highlightbackground=ENTRY_BORDER,
            highlightcolor=ACCENT, buttonbackground=PANEL,
            font=(FONT_FAMILY, 11, "bold"), justify=tk.CENTER,
            validate="key",
            validatecommand=(self.register(
                self._is_float_or_empty if is_float else self._is_int_or_empty
            ), "%P"))
        spin.pack(ipady=4)
        if not hasattr(self, "_widget_refs"):
            self._widget_refs = []
        self._widget_refs.append(spin)

    def _sync_f2(self):
        try:
            f1 = int(float(self.params["f1"].get()))
            self.f2_var.set(str(f1 * 2))
        except Exception:
            pass

    # ====================== CSV 工具：预览 / 导出 ====================== #
    def _preview_csv(self):
        """打开 Toplevel，预览单个脑电 CSV：元信息 + 12 通道前 2 秒波形。"""
        path = filedialog.askopenfilename(
            title="选择脑电 CSV 文件",
            initialdir=self.move_var.get() or BASE_DIR,
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")])
        if not path:
            return

        try:
            info = csv_summary(path, channels=CHANNELS)
            import pandas as _pd
            df = _pd.read_csv(path)
            sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0)
            if sig.shape[1] > CHANNELS:
                sig = sig[:, :CHANNELS]
            elif sig.shape[1] < CHANNELS:
                pad = np.zeros((sig.shape[0], CHANNELS - sig.shape[1]),
                               dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=1)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return

        win = tk.Toplevel(self)
        win.title("预览 CSV · %s" % os.path.basename(path))
        win.geometry("780x620")
        win.configure(bg=BG)

        # 顶部元信息
        head = tk.Frame(win, bg=PANEL, bd=1, relief=tk.FLAT,
                        highlightthickness=1,
                        highlightbackground=PANEL_BORDER)
        head.pack(fill=tk.X, padx=12, pady=(12, 6))
        tk.Label(head, text="文件：%s" % os.path.basename(path),
                 bg=PANEL, fg=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(
            anchor=tk.W, padx=12, pady=(8, 2))
        meta_lines = [
            "路径：%s" % path,
            "行数：%d   时长：%.2f 秒（按 128 Hz 估算）" % (
                info["rows"], info["duration_s"]),
            "通道数：%d   信号质量均值：%.1f   NaN 数：%d" % (
                len(info["per_channel"]), info["quality_mean"],
                sum(n for _, _, _, _, _, n in info["per_channel"])),
        ]
        for s in meta_lines:
            tk.Label(head, text=s, bg=PANEL, fg=TXT2,
                     font=(FONT_FAMILY, 9)).pack(anchor=tk.W, padx=12, pady=0)

        # 通道统计表（简化为 Text）
        stat_box = tk.Frame(win, bg=PANEL, bd=1, relief=tk.FLAT,
                            highlightthickness=1,
                            highlightbackground=PANEL_BORDER)
        stat_box.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(stat_box, text="通道统计（前 %d 行采样）" % min(200000, info["rows"]),
                 bg=PANEL, fg=ACCENT, font=(FONT_FAMILY, 10, "bold")).pack(
            anchor=tk.W, padx=12, pady=(6, 2))
        st = tk.Text(stat_box, height=8, bg="#FFFFFF", fg=TXT, bd=0,
                     font=("Consolas", 9))
        st.pack(fill=tk.X, padx=12, pady=(0, 8))
        st.insert(tk.END,
                  "%-12s %10s %10s %10s %10s %6s\n" % (
                      "channel", "min", "max", "mean", "std", "nan"))
        st.insert(tk.END, "-" * 64 + "\n")
        for name, vmin, vmax, vmean, vstd, nn in info["per_channel"]:
            st.insert(tk.END,
                      "%-12s %10.2f %10.2f %10.2f %10.2f %6d\n" % (
                          name, vmin, vmax, vmean, vstd, nn))
        st.config(state=tk.DISABLED)

        # 波形（前 2 秒 = 前 256 点；通道数 12 -> 画 4×3 子图过密，
        # 改为画前 6 个最常用通道）
        fig = Figure(figsize=(7.4, 3.0), dpi=100, facecolor=BG)
        plot_chs = list(range(min(6, sig.shape[1])))
        names = [info["per_channel"][i][0] for i in plot_chs]
        for i, ch in enumerate(plot_chs):
            ax = fig.add_subplot(2, 3, i + 1, facecolor="#FFFFFF")
            n_show = min(256, sig.shape[0])
            ax.plot(sig[:n_show, ch], color=ACCENT, linewidth=1.0)
            ax.set_title(names[ch], fontsize=8, color=TXT)
            ax.tick_params(colors=TXT2, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(PANEL_BORDER)
            ax.grid(True, color="#F0F0F0", linewidth=0.5)
        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                    padx=12, pady=(4, 12))

    def _export_dataset(self):
        """弹出导出对话框：把 move+stay 合并为单个 CSV（带 label）。"""
        path = filedialog.asksaveasfilename(
            title="导出数据集为 CSV",
            initialdir=self.export_var.get() or BASE_DIR,
            initialfile="dataset_merged.csv",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            opts = self._collect_preprocess_opts()
            fs = 128.0
            if hasattr(self, "preproc"):
                try:
                    fs = float(self.preproc["fs"].get() or "128")
                except Exception:
                    fs = 128.0
            self.log_msg("导出数据集 → %s …" % path, "info")
            self.log_msg(self._preprocess_summary_text(opts), "info")
            nm, ns, tot = export_dataset_csv(
                self.move_var.get(), self.stay_var.get(), path,
                channels=CHANNELS, preprocess_opts=opts, fs=fs)
            self.log_msg("导出完成：move=%d  stay=%d  共 %d 行" % (
                nm, ns, tot), "ok")
            messagebox.showinfo(
                "导出完成",
                "已写入：%s\nmove=%d  stay=%d  合计 %d 行" % (path, nm, ns, tot))
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
            self.log_msg("导出失败：%s" % e, "err")

    # ====================== 右侧：大曲线区 + 日志条 ====================== #
    def _build_plot_area(self, parent):
        # 底部日志条（固定高度，先 pack 保证可见）
        log_box = tk.Frame(parent, bg=PANEL, bd=1, relief=tk.FLAT,
                           highlightthickness=1,
                           highlightbackground=PANEL_BORDER, height=140)
        log_box.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        log_box.pack_propagate(False)

        tk.Label(log_box, text="训练日志", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 10, "bold")).pack(anchor=tk.W,
                                                      padx=12, pady=(6, 2))
        self.log = tk.Text(log_box, bg="#FFFFFF", fg=TXT, bd=0,
                           highlightthickness=1,
                           highlightbackground=PANEL_BORDER,
                           font=("Consolas", 9), wrap=tk.WORD,
                           state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        # 曲线区（占据剩余全部空间）
        plot_box = tk.Frame(parent, bg=PANEL, bd=1, relief=tk.FLAT,
                            highlightthickness=1,
                            highlightbackground=PANEL_BORDER)
        plot_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tk.Label(plot_box, text="训练曲线", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(anchor=tk.W,
                                                      padx=12, pady=(8, 0))

        self.fig = Figure(figsize=(8, 5.6), dpi=100, facecolor=PANEL)
        self.ax_loss = self.fig.add_subplot(211, facecolor="#FFFFFF")
        self.ax_acc = self.fig.add_subplot(212, facecolor="#FFFFFF")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=12, pady=(4, 10))
        self._style_axes()
        self.fig.tight_layout(pad=2.0)
        self._draw_empty()

    # ====================== 样式 ====================== #
    def _style_axes(self):
        for ax, title, color in [
                (self.ax_loss, "Loss 下降", AMBER),
                (self.ax_acc, "准确率 上升", GREEN)]:
            ax.set_title(title, color=TXT, fontname=FONT_FAMILY, fontsize=11)
            ax.set_xlabel("Epoch", color=TXT2, fontname=FONT_FAMILY, fontsize=9)
            ax.tick_params(colors=TXT2)
            for sp in ax.spines.values():
                sp.set_color(PANEL_BORDER)
            ax.grid(True, color="#F0F0F0", linewidth=0.6)

    def _draw_empty(self):
        self.ax_loss.clear(); self.ax_acc.clear()
        self._style_axes()
        self.ax_loss.text(0.5, 0.5, "等待训练…", color=GRAY, ha="center",
                          va="center", fontname=FONT_FAMILY, fontsize=12)
        self.ax_acc.text(0.5, 0.5, "等待训练…", color=GRAY, ha="center",
                         va="center", fontname=FONT_FAMILY, fontsize=12)
        self.canvas.draw()

    # ====================== 校验辅助 ====================== #
    def _is_int_or_empty(self, s):
        if s == "":
            return True
        try:
            int(s)
            return True
        except ValueError:
            return False

    def _is_float_or_empty(self, s):
        if s == "":
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    # ====================== 数据集信息 ====================== #
    def _refresh_dataset_info(self):
        try:
            win = int(float(self.params["window"].get()))
            stp = int(float(self.params["stride"].get()))
        except Exception:
            self.info_var.set("窗口/步长无效，请检查输入。")
            return
        try:
            info = dataset_summary(self.move_var.get(), self.stay_var.get(),
                                   win, stp)
            mw = sum(w for _, _, w in info["move_files"])
            sw = sum(w for _, _, w in info["stay_files"])
            txt = ("数据集：\n  运动 %d 文件 / %d 窗\n  静止 %d 文件 / %d 窗\n"
                   "  合计 %d 训练窗口" % (
                       len(info["move_files"]), mw,
                       len(info["stay_files"]), sw, mw + sw))
        except Exception as e:
            txt = "数据集读取失败：%s" % e
        self.info_var.set(txt)

    # ====================== 预处理参数采集 ====================== #
    def _collect_preprocess_opts(self):
        """从 self.preproc 收集当前勾选与参数，返回 dict 或 None（全关时返回 None）。"""
        if not hasattr(self, "preproc"):
            return None
        p = self.preproc
        opts = {}
        if p["detrend"].get():
            opts["detrend"] = True
        if p["bandpass"].get():
            opts["bandpass"] = True
            opts["low"] = float(p["low"].get() or "1.0")
            opts["high"] = float(p["high"].get() or "40.0")
        if p["notch"].get():
            opts["notch"] = True
            opts["notch_freq"] = float(p["notch_freq"].get() or "50")
            opts["notch_q"] = float(p["notch_q"].get() or "30")
        if p["clip"].get():
            opts["clip"] = True
            opts["clip_sigma"] = float(p["clip_sigma"].get() or "3.5")
        # signalQuality（通道 0）跳过带通/陷波/裁剪（在 data_utils 中实现）
        return opts if opts else None

    def _preprocess_summary_text(self, opts):
        """把 opts 翻译成中文一行字符串，用于配置确认区。"""
        if not opts:
            return "预处理：无"
        parts = []
        if opts.get("detrend"):
            parts.append("去基线")
        if opts.get("bandpass"):
            parts.append("带通 %.1f-%.1fHz" % (opts["low"], opts["high"]))
        if opts.get("notch"):
            parts.append("陷波 %.0fHz" % opts["notch_freq"])
        if opts.get("clip"):
            parts.append("裁剪 %.1fσ" % opts["clip_sigma"])
        return "预处理：" + " + ".join(parts)

    # ====================== 参数校验 ====================== #
    def _validate_config(self):
        try:
            epochs = int(float(self.params["epochs"].get()))
        except Exception:
            return None, "轮次 epochs 必须为整数。"
        if not (1 <= epochs <= 9999):
            return None, "轮次 epochs 需在 1 ~ 9999 之间。"

        raw = {
            "lr": (self.params["lr"].get(), True, 1e-4, 10.0, "学习率 lr"),
            "f1": (self.params["f1"].get(), False, 1, 64, "卷积核数 f1"),
            "k1": (self.params["k1"].get(), False, 2, 15, "核尺寸 k1"),
            "k2": (self.params["k2"].get(), False, 2, 11, "核尺寸 k2"),
            "batch": (self.params["batch"].get(), False, 0, 1024, "批大小"),
            "window": (self.params["window"].get(), False, 32, 1024, "窗口大小"),
            "stride": (self.params["stride"].get(), False, 1, 256, "滑窗步长"),
        }
        cfg = {"epochs": epochs}
        for key, (val, is_float, lo, hi, name) in raw.items():
            try:
                v = float(val) if is_float else int(float(val))
            except Exception:
                return None, "%s 必须为数字。" % name
            if v < lo or v > hi:
                return None, "%s 需在 %s ~ %s 之间。" % (name, lo, hi)
            cfg[key] = v
        cfg["lr"] = float(cfg["lr"])
        cfg["f2"] = int(cfg["f1"]) * 2   # 自动 = 2 × f1
        if cfg["k1"] >= cfg["window"] or cfg["k2"] >= (cfg["window"] // 2):
            return None, "核尺寸必须小于窗口（k1<window 且 k2<window/2）。"

        for label, var in [("运动数据目录", self.move_var),
                           ("静止数据目录", self.stay_var),
                           ("导出目录", self.export_var)]:
            if not var.get().strip():
                return None, "%s 不能为空。" % label
        cfg["_move_dir"] = self.move_var.get().strip()
        cfg["_stay_dir"] = self.stay_var.get().strip()
        cfg["_export_dir"] = self.export_var.get().strip()
        return cfg, None

    def _config_text(self, cfg, samples):
        return ("epochs = %d   lr = %.3g\n"
                "f1 = %d  f2 = %d (自动 = 2×f1)  k1 = %d  k2 = %d\n"
                "batch = %d  window = %d  stride = %d\n"
                "训练样本 = %d" % (
                    cfg["epochs"], cfg["lr"], cfg["f1"], cfg["f2"],
                    cfg["k1"], cfg["k2"], cfg["batch"],
                    cfg["window"], cfg["stride"], samples))

    # ====================== 训练控制 ====================== #
    def on_start(self):
        if self.training:
            return
        cfg, err = self._validate_config()
        if err:
            messagebox.showerror("参数错误", err)
            return
        move_dir = cfg.pop("_move_dir")
        stay_dir = cfg.pop("_stay_dir")
        export_dir = cfg.pop("_export_dir")

        for label, d in [("运动数据目录", move_dir), ("静止数据目录", stay_dir)]:
            if not os.path.isdir(d):
                messagebox.showerror("数据目录错误", "%s 不存在：\n%s" % (label, d))
                return
            if not any(fn.lower().endswith(".csv") for fn in os.listdir(d)):
                messagebox.showerror("数据目录错误",
                                      "%s 下未找到 CSV 文件：\n%s" % (label, d))
                return

        self._refresh_dataset_info()
        total = cfg["epochs"]
        self.epochs_log = []
        self._draw_empty()
        self.prog_var.set(0)
        self.prog_lbl.config(text="0 / %d" % total)

        self.conf_var.set(self._config_text(cfg, 0))
        self._set_controls_enabled(False)
        self._set_status("训练中…", "warn")

        self.training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.train_start_ts = time.time()
        self.last_epoch_ts = time.time()

        # 清理残留停止信号（防止上一轮 set() 后粘性 Event 影响本轮）
        self.stop_event.clear()

        pp_opts = self._collect_preprocess_opts()

        t = threading.Thread(target=self._train_worker,
                              args=(cfg, move_dir, stay_dir, export_dir,
                                    pp_opts),
                              daemon=True)
        t.start()

    def on_stop(self):
        if not self.training:
            return
        self.stop_event.set()
        self.log_msg("已请求停止，当前轮结束后终止…", "warn")
        self._set_status("停止中…", "warn")

    def _train_worker(self, cfg, move_dir, stay_dir, export_dir,
                       preprocess_opts=None):
        try:
            self.log_msg("加载数据：move=%s" % os.path.basename(move_dir), "info")
            self.log_msg("加载数据：stay=%s" % os.path.basename(stay_dir), "info")
            self.log_msg(self._preprocess_summary_text(preprocess_opts), "info")
            fs = 128.0
            if preprocess_opts and "fs" in self.preproc:
                try:
                    fs = float(self.preproc["fs"].get() or "128")
                except Exception:
                    fs = 128.0
            X, y = load_dataset(move_dir, stay_dir,
                                window=int(cfg["window"]),
                                stride=int(cfg["stride"]),
                                channels=CHANNELS,
                                preprocess_opts=preprocess_opts, fs=fs)
            self.log_msg("样本数=%d  运动=%d  静止=%d" % (
                X.shape[0], int((y == 1).sum()), int((y == 0).sum())), "ok")
            self.queue.put(("samples", X.shape[0]))

            self.scaler = Scaler().fit(X)
            Xn = self.scaler.transform(X)

            self.model = EEGCNN(f1=int(cfg["f1"]), f2=int(cfg["f2"]),
                                k1=int(cfg["k1"]), k2=int(cfg["k2"]),
                                channels=CHANNELS, window=int(cfg["window"]))
            total = cfg["epochs"]

            def on_epoch(ep, loss, acc, lr):
                if self.stop_event.is_set():
                    return
                self.queue.put(("epoch", ep, loss, acc, total))

            self.model.fit(Xn, y, lr=cfg["lr"], epochs=cfg["epochs"],
                           batch_size=int(cfg["batch"]), on_epoch=on_epoch,
                           stop_flag=self.stop_event.is_set)

            weights = self.model.export(export_dir, self.scaler)
            self.queue.put(("done", export_dir, X.shape[0]))
        except Exception as e:
            import traceback
            self.queue.put(("error", str(e), traceback.format_exc()))

    # ====================== 队列刷新 ====================== #
    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        try:
            self._poll_id = self.after(80, self._poll_queue)
        except Exception:
            pass

    def destroy(self):
        """销毁时取消 after 回调，避免 dangling tk 错误。"""
        try:
            if getattr(self, "_poll_id", None):
                self.after_cancel(self._poll_id)
                self._poll_id = None
        except Exception:
            pass
        super().destroy()

    def _handle_msg(self, msg):
        kind = msg[0]
        if kind == "samples":
            _, n = msg
            cur = self.conf_var.get()
            self.conf_var.set(cur.split("\n训练样本")[0] +
                              "\n训练样本 = %d" % n)
        elif kind == "epoch":
            _, ep, loss, acc, total = msg
            now = time.time()
            cost = now - self.last_epoch_ts
            self.last_epoch_ts = now
            self.epochs_log.append((ep, loss, acc))
            self.prog_var.set(100.0 * (ep + 1) / max(total, 1))
            eta = cost * (total - ep - 1)
            self.prog_lbl.config(
                text="%d / %d   loss=%.4f  acc=%.3f  ETA %s" % (
                    ep + 1, total, loss, acc, self._fmt_time(eta)))
            if ep % REDRAW_EVERY == 0 or ep == total - 1:
                self._redraw_curves()
            if ep % 10 == 0 or ep == total - 1:
                self.log_msg("epoch %3d  loss=%.4f  acc=%.3f  (%.1fs)" % (
                    ep, loss, acc, cost), "info")
        elif kind == "done":
            _, export_dir, n = msg
            self._redraw_curves()
            if len(self.epochs_log) == 0:
                # 防御：正常情况不应发生；若出现说明被瞬时停止或残留信号干扰
                self.log_msg(
                    "警告：本轮训练未记录任何 epoch（可能瞬时停止或停止信号残留）",
                    "err")
            self.log_msg("训练完成，已导出权重到：%s" % export_dir, "ok")
            self._set_status("训练完成", "ok")
            self._finish()
        elif kind == "error":
            _, err, tb = msg
            self.log_msg("训练出错：%s" % err, "err")
            self.log_msg(tb, "err")
            self._set_status("训练出错", "err")
            self._finish()

    def _finish(self):
        self.training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_controls_enabled(True)
        self._set_status("就绪", "info")
        # 清理 Event 残留，避免下次启动时模型在 ep=0 立刻 break（0 轮 bug）
        self.stop_event.clear()

    def _set_controls_enabled(self, enabled):
        st = tk.NORMAL if enabled else tk.DISABLED
        for w in self._widget_refs:
            try:
                w.config(state=st)
            except Exception:
                pass

    # ====================== 辅助 ====================== #
    def _redraw_curves(self):
        if not self.epochs_log:
            return
        eps = [e for e, _, _ in self.epochs_log]
        losses = [l for _, l, _ in self.epochs_log]
        accs = [a for _, _, a in self.epochs_log]
        self.ax_loss.clear(); self.ax_acc.clear()
        self._style_axes()
        self.ax_loss.plot(eps, losses, color=AMBER, linewidth=1.6)
        self.ax_acc.plot(eps, accs, color=GREEN, linewidth=1.6)
        if len(eps) > 1:
            self.ax_loss.scatter([eps[-1]], [losses[-1]], color=AMBER, s=14)
            self.ax_acc.scatter([eps[-1]], [accs[-1]], color=GREEN, s=14)
            self.ax_acc.set_ylim(min(0.0, min(accs) - 0.05), 1.02)
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def log_msg(self, text, level="info"):
        """追加一行日志。子线程调用时通过 after 调度到主线程，避免
        Tk 在 Python 3.14+ 报 'main thread is not in main loop'。"""
        def _do():
            try:
                color = {"info": TXT2, "ok": GREEN, "warn": AMBER,
                         "err": RED}.get(level, TXT2)
                self.log.config(state=tk.NORMAL)
                self.log.insert(tk.END, text + "\n", level)
                self.log.tag_config(level, foreground=color)
                self.log.see(tk.END)
                self.log.config(state=tk.DISABLED)
            except Exception:
                pass
        try:
            self.after(0, _do)
        except Exception:
            # 主线程或控件已销毁：直接执行（不开新线程则安全）
            _do()


def main():
    root = tk.Tk()
    root.title("eeggamev1.0 · AI 训练程序")
    root.geometry("1180x800")
    root.configure(bg=BG)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("EEG.Horizontal.TProgressbar",
                    troughcolor="#F3F4F6", background=ACCENT, thickness=14)
    panel = TrainPanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()