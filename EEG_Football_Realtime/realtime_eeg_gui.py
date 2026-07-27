# -*- coding: utf-8 -*-
"""
实时脑电检测控制台（Tkinter + matplotlib）

功能：
  - 选择 TGAM 串口（COM）与波特率，连接后实时读取 RAW 脑电
  - 选择模型文件（默认 eeg_cnn_best_weights.npy）
  - 实时滑窗推理：单通道广播 12 通道 -> EEGCNN -> 0/1 运动/静止
  - 实时覆盖写入 movda.txt（驱动足球游戏 server.js -> 红方前进）
  - 可视化：RAW 实时波形、运动/静止概率柱状图、红/绿信号灯
  - 无传感器时支持「离线 CSV 回放」验证闭环
  - 一键「启动游戏」：打开 server.js 并启动浏览器

界面/文件统一 UTF-8 与 SimHei 中文字体（深色控制台风格）。
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque

import numpy as np

# matplotlib 后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

import tgam_serial
import realtime_core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = os.path.join(BASE_DIR, "eeg_cnn_best_weights.npy")
MOVDA_PATH = os.path.join(BASE_DIR, "movda.txt")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

# ----------------------- 字体 / 配色（深色控制台） ----------------------- #
FONT = ("SimHei", 11)
FONT_TITLE = ("SimHei", 18, "bold")
FONT_SUB = ("SimHei", 13)
FONT_BTN = ("SimHei", 12)

BG = "#1E1E1E"
BG2 = "#252526"
FG = "#FFFFFF"
FG_DIM = "#D0D0D0"
PRIMARY = "#2D7FF9"
PRIMARY2 = "#1E88E5"
RED = "#E53935"
GREEN = "#43A047"

# matplotlib 中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False


class ConsoleApp:
    def __init__(self, root):
        self.root = root
        root.title("实时脑电检测控制台 · TGAM")
        root.configure(bg=BG)
        root.geometry("1024x720")
        try:
            root.state("zoomed")
        except Exception:
            pass

        self.reader = None
        self.inferencer = None
        self.running = False
        self.server_proc = None

        # 实时数据缓冲（供绘图）
        self.plot_buf = deque(maxlen=realtime_core.WINDOW_SIZE * 2)
        self.prob_move = 0.5
        self.pred = 0

        self._build_ui()
        self._init_inferencer()
        self._refresh_ports()

        # 定时刷新绘图
        self._anim = FuncAnimation(self.fig, self._update_plot, interval=100,
                                   blit=False, cache_frame_data=False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # 顶部标题
        frm_top = tk.Frame(self.root, bg=BG)
        frm_top.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(frm_top, text="实时脑电检测控制台", font=FONT_TITLE,
                 fg=PRIMARY, bg=BG).pack(side="left")
        tk.Label(frm_top, text="TGAM 串口协议 · 运动/静止实时判定",
                 font=FONT_SUB, fg=FG_DIM, bg=BG).pack(side="left", padx=12)

        # 连接控制区
        frm_ctrl = tk.Frame(self.root, bg=BG2, bd=1, relief="groove")
        frm_ctrl.pack(fill="x", padx=16, pady=6)

        tk.Label(frm_ctrl, text="串口:", font=FONT, fg=FG, bg=BG2).grid(
            row=0, column=0, padx=8, pady=8, sticky="e")
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(frm_ctrl, textvariable=self.port_var,
                                    font=FONT, width=12, state="readonly")
        self.port_cb.grid(row=0, column=1, padx=4, pady=8)
        tk.Button(frm_ctrl, text="刷新", font=FONT_BTN, bg="#3A3A3A", fg=FG,
                  command=self._refresh_ports).grid(row=0, column=2, padx=4, pady=8)

        tk.Label(frm_ctrl, text="波特率:", font=FONT, fg=FG, bg=BG2).grid(
            row=0, column=3, padx=(16, 4), pady=8, sticky="e")
        self.baud_var = tk.StringVar(value="57600")
        tk.Entry(frm_ctrl, textvariable=self.baud_var, font=FONT, width=8,
                 bg="#333", fg=FG, insertbackground=FG).grid(
            row=0, column=4, padx=4, pady=8)

        tk.Label(frm_ctrl, text="模型:", font=FONT, fg=FG, bg=BG2).grid(
            row=1, column=0, padx=8, pady=8, sticky="e")
        self.model_var = tk.StringVar(value=MODEL_DEFAULT)
        tk.Entry(frm_ctrl, textvariable=self.model_var, font=FONT, width=46,
                 bg="#333", fg=FG, insertbackground=FG).grid(
            row=1, column=1, columnspan=3, padx=4, pady=8, sticky="w")
        tk.Button(frm_ctrl, text="浏览", font=FONT_BTN, bg="#3A3A3A", fg=FG,
                  command=self._browse_model).grid(
            row=1, column=4, padx=4, pady=8)

        # 按钮区
        frm_btn = tk.Frame(self.root, bg=BG)
        frm_btn.pack(fill="x", padx=16, pady=4)
        self.btn_connect = tk.Button(frm_btn, text="▶ 连接传感器", font=FONT_BTN,
                                     bg=PRIMARY, fg="white", width=14,
                                     command=self.toggle_connect)
        self.btn_connect.pack(side="left", padx=4)
        self.btn_replay = tk.Button(frm_btn, text="▶ 离线CSV回放", font=FONT_BTN,
                                    bg="#3A3A3A", fg=FG, width=14,
                                    command=self.run_replay)
        self.btn_replay.pack(side="left", padx=4)
        self.btn_game = tk.Button(frm_btn, text="🎮 启动游戏", font=FONT_BTN,
                                  bg=GREEN, fg="white", width=14,
                                  command=self.launch_game)
        self.btn_game.pack(side="left", padx=4)
        self.btn_stop = tk.Button(frm_btn, text="■ 停止", font=FONT_BTN,
                                  bg=RED, fg="white", width=10, state="disabled",
                                  command=self.stop_all)
        self.btn_stop.pack(side="left", padx=4)

        # 可视化区
        frm_viz = tk.Frame(self.root, bg=BG)
        frm_viz.pack(fill="both", expand=True, padx=16, pady=6)

        self.fig = plt.Figure(figsize=(9, 4.6), dpi=100)
        self.fig.patch.set_facecolor(BG)
        self.ax_wave = self.fig.add_subplot(2, 1, 1)
        self.ax_prob = self.fig.add_subplot(2, 1, 2)
        self._init_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm_viz)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # 信号灯 + 状态
        frm_light = tk.Frame(self.root, bg=BG2, bd=1, relief="groove")
        frm_light.pack(fill="x", padx=16, pady=6)
        self.light = tk.Canvas(frm_light, width=60, height=60, bg=BG2,
                               highlightthickness=0)
        self.light.pack(side="left", padx=12, pady=8)
        self._light_id = self.light.create_oval(8, 8, 52, 52, fill=GREEN,
                                                outline="")
        self.status_var = tk.StringVar(value="状态：未连接 · 等待信号")
        tk.Label(frm_light, textvariable=self.status_var, font=FONT_SUB,
                 fg=FG, bg=BG2).pack(side="left", padx=8)

        # 日志
        frm_log = tk.Frame(self.root, bg=BG)
        frm_log.pack(fill="both", padx=16, pady=(0, 10))
        self.log = tk.Text(frm_log, height=7, bg="#161616", fg="#9CDCFE",
                           font=("Consolas", 10), insertbackground=FG)
        self.log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ #
    def _init_axes(self):
        for ax in (self.ax_wave, self.ax_prob):
            ax.set_facecolor(BG)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=FG_DIM)
            ax.xaxis.label.set_color(FG_DIM)
            ax.yaxis.label.set_color(FG_DIM)
            ax.title.set_color(FG)
        self.ax_wave.set_title("实时脑电 RAW 波形", fontsize=12)
        self.ax_wave.set_ylabel("幅值", fontsize=10)
        self.ax_prob.set_title("运动 / 静止 概率", fontsize=12)
        self.ax_prob.set_ylabel("概率", fontsize=10)
        self.ax_prob.set_ylim(0, 1)
        self.bars = self.ax_prob.bar(["静止", "运动"], [0.5, 0.5],
                                     color=[GREEN, RED])
        self.fig.tight_layout()

    def _init_inferencer(self):
        self.inferencer = realtime_core.RealtimeInference(
            model_path=self.model_var.get(),
            on_result=self._on_result,
            on_log=self.log_msg)
        try:
            self.inferencer.load_model()
        except Exception as e:
            self.log_msg("模型加载失败: %s" % e, "err")

    # ------------------------------------------------------------------ #
    def _refresh_ports(self):
        ports = tgam_serial.list_ports() or []
        if not ports:
            ports = ["（无串口，可用离线回放）"]
        self.port_cb["values"] = ports
        if ports and ports[0].startswith("（"):
            self.port_var.set(ports[0])
        else:
            self.port_var.set(ports[0])

    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="选择模型权重文件", initialdir=BASE_DIR,
            filetypes=[("NumPy 权重", "*.npy"), ("全部", "*.*")])
        if p:
            self.model_var.set(p)
            self._init_inferencer()

    # ------------------------------------------------------------------ #
    def toggle_connect(self):
        if self.reader and self.reader.is_open():
            self.stop_all()
            return
        port = self.port_var.get()
        if port.startswith("（"):
            messagebox.showwarning("提示", "未检测到串口，请使用「离线CSV回放」模式，"
                                            "或检查传感器连接。")
            return
        try:
            baud = int(self.baud_var.get())
        except ValueError:
            baud = 57600
        self.reader = tgam_serial.TGAMReader(
            port=port, baud=baud,
            on_raw=self._on_raw, on_packet=self._on_packet,
            on_status=self.log_msg)
        try:
            self.reader.open()
        except Exception as e:
            messagebox.showerror("连接失败", str(e))
            return
        self.running = True
        self.inferencer.reset()
        self.btn_connect.config(text="■ 断开", bg=RED)
        self.btn_stop.config(state="normal")
        self.set_status("已连接 · 实时检测中…")

    def _on_raw(self, value):
        """RAW 样本（约 512Hz）：用于波形展示 + RAW 广播模式回退。"""
        if not self.running:
            return
        self.inferencer.add_raw(value)
        self.plot_buf.append(value)

    def _on_packet(self, packet):
        """TGAM 大包：构造 12 通道特征向量喂入模型（实时首选路径）。"""
        if not self.running:
            return
        vec = tgam_serial.TGAMReader.build_feature_vector(packet)
        self.inferencer.add_feature_vector(vec)

    def _on_result(self, pred, prob_move, raw_buf):
        self.pred = pred
        self.prob_move = prob_move
        color = RED if pred == 1 else GREEN
        label = "运动 (1) → 红方前进" if pred == 1 else "静止 (0) → 停止"
        self.root.after(0, lambda: self._set_light(color, label))

    # ------------------------------------------------------------------ #
    def run_replay(self):
        path = filedialog.askopenfilename(
            title="选择脑电 CSV（离线回放）", initialdir=SAMPLE_DIR,
            filetypes=[("CSV", "*.csv"), ("全部", "*.*")])
        if not path:
            return
        import pandas as pd
        import time
        try:
            df = pd.read_csv(path)
            sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0)[:, :realtime_core.CHANNELS]
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return
        n_rows = sig.shape[0]
        if n_rows < realtime_core.WINDOW_SIZE:
            messagebox.showwarning("样本不足",
                                    "该 CSV 样本数 < 128，无法进行滑窗推理。"
                                    "请选择更长的脑电记录。")
            return
        self.stop_all()
        self.running = False
        self.inferencer.reset()
        self.log_msg("离线回放: %s（%d 行 × %d 通道）"
                     % (os.path.basename(path), n_rows, realtime_core.CHANNELS),
                     "info")
        self.btn_replay.config(text="■ 回放中…", bg=RED)

        def worker():
            stride = 2
            total_win = (n_rows - realtime_core.WINDOW_SIZE) // stride + 1
            for idx, s in enumerate(range(0, n_rows - realtime_core.WINDOW_SIZE + 1,
                                           stride)):
                w = sig[s:s + realtime_core.WINDOW_SIZE]  # (128, 12)
                # 取第一通道（signalQuality）推入波形绘图缓冲
                for v in w[:, 0]:
                    self.plot_buf.append(float(v))
                # 滑窗推理并实时写 movda.txt（write=True 控制游戏）
                self.inferencer.infer_window(w)
                if idx % 10 == 0:
                    pct = 100.0 * idx / max(total_win, 1)
                    self.root.after(0, lambda p=pct: self.set_status(
                        "回放中 %.0f%%" % p))
                time.sleep(0.03)  # 模拟实时节奏
            self.root.after(0, lambda: self.set_status("回放完成 · 12通道窗口推理"))
            self.root.after(0, lambda: self.btn_replay.config(
                text="▶ 离线CSV回放", bg="#3A3A3A"))
            self.log_msg("离线回放结束（%d 窗）。" % total_win, "ok")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    def launch_game(self):
        if self.server_proc and self.server_proc.poll() is None:
            self.log_msg("游戏服务器已在运行。", "warn")
            self._open_browser()
            return
        try:
            self.server_proc = subprocess.Popen(
                ["node", "server.js"], cwd=BASE_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.log_msg("已启动游戏服务器 (node server.js)。", "ok")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return
        self.root.after(1200, self._open_browser)

    def _open_browser(self):
        url = "http://localhost:8080"
        try:
            import webbrowser
            webbrowser.open(url)
            self.log_msg("已打开游戏页面: %s" % url, "ok")
        except Exception:
            self.log_msg("请手动打开: %s" % url, "info")

    # ------------------------------------------------------------------ #
    def stop_all(self):
        self.running = False
        if self.reader:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
        self.btn_connect.config(text="▶ 连接传感器", bg=PRIMARY)
        self.btn_stop.config(state="disabled")
        self.set_status("已停止")
        self.root.after(0, lambda: self._set_light(GREEN, "静止 (0) → 停止"))

    # ------------------------------------------------------------------ #
    def _set_light(self, color, label):
        self.light.itemconfig(self._light_id, fill=color,
                              outline=color)
        self.light.configure(bg=BG2)
        self.set_status(label)

    def set_status(self, text):
        self.status_var.set("状态：" + text)

    def log_msg(self, msg, level="info"):
        color = {"ok": "#43A047", "err": "#E53935", "warn": "#FFB300",
                 "info": "#9CDCFE"}.get(level, "#9CDCFE")
        try:
            self.log.insert("end", msg + "\n", level)
            self.log.tag_config(level, foreground=color)
            self.log.see("end")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _update_plot(self, frame=None):
        if self.plot_buf:
            arr = np.array(self.plot_buf, dtype=np.float32)
            self.ax_wave.clear()
            self.ax_wave.set_facecolor(BG)
            self.ax_wave.plot(arr, color=PRIMARY, linewidth=1.0)
            self.ax_wave.set_title("实时脑电 RAW 波形", fontsize=12, color=FG)
            self.ax_wave.set_ylabel("幅值", fontsize=10, color=FG_DIM)
            self.ax_wave.tick_params(colors=FG_DIM)
            for s in ("top", "right"):
                self.ax_wave.spines[s].set_visible(False)
        # 概率柱状图
        self.bars[0].set_height(1 - self.prob_move)
        self.bars[1].set_height(self.prob_move)
        self.ax_prob.set_title("运动 / 静止 概率", fontsize=12, color=FG)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def _on_close(self):
        self.stop_all()
        if self.server_proc and self.server_proc.poll() is None:
            try:
                self.server_proc.terminate()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConsoleApp(root)
    root.mainloop()
