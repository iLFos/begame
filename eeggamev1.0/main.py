# -*- coding: utf-8 -*-
"""
main.py

eeggamev1.0 入口：单窗口 Tkinter 应用，顶部下拉切换
「AI 训练程序」与「AI 推理检测程序」两个面板。

风格：白色简约。状态栏统一展示各模块运行状态。退出时统一回收
node 子进程、TGAM 串口、matplotlib 画布。

顶部栏新增「📁 数据目录」入口：紧凑 chip 显示当前 3 个路径摘要，
点击 ⚙ 弹出 Toplevel 模态编辑并保存到 eeg_config.json。
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 让同目录下的模块可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_store
from train_panel import TrainPanel
from infer_panel import InferPanel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


def _abbreviate(path, max_len=18):
    """把长路径缩成「…/末段」形式，便于 chip 内显示。"""
    if not path:
        return "(未设置)"
    p = path.replace("\\", "/")
    if len(p) <= max_len:
        return p
    base = os.path.basename(p.rstrip("/"))
    parent = os.path.basename(os.path.dirname(p.rstrip("/")))
    if parent and (len(parent) + 1 + len(base) <= max_len):
        return ".../%s/%s" % (parent, base)
    return ".../" + base[-max_len:]


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("eeggamev1.0 · 脑电 AI 训练与推理控制")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = min(1280, sw - 80)
        h = min(860, sh - 100)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 20)
        self.root.geometry("%dx%d+%d+%d" % (w, h, x, y))
        self.root.minsize(1000, 620)
        self.root.configure(bg=BG)

        # 全局配置加载（三个字段都为空时回退到 train_panel 默认值，
        # 让顶部 chip 显示有意义的路径）
        self.cfg = config_store.load_config()
        if not any(self.cfg.get(k) for k in ("move_dir", "stay_dir", "export_dir")):
            try:
                from train_panel import (DEFAULT_MOVE_DIR, DEFAULT_STAY_DIR,
                                          DEFAULT_EXPORT_DIR)
                self.cfg = {
                    "move_dir": DEFAULT_MOVE_DIR,
                    "stay_dir": DEFAULT_STAY_DIR,
                    "export_dir": DEFAULT_EXPORT_DIR,
                }
            except Exception:
                pass

        # 关闭钩子
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 面板实例
        self.train_panel = None
        self.infer_panel = None
        self.paths_dialog = None

        self._build_topbar()
        self._build_body()
        self._build_statusbar()

        # 初始化面板
        self._create_panels()
        self._switch_panel("train")
        self._update_paths_chip()

    # ====================== 顶部栏 ====================== #
    def _build_topbar(self):
        top = tk.Frame(self.root, bg=BG, bd=0, highlightthickness=0)
        top.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(16, 4))

        # 左：标题
        title_box = tk.Frame(top, bg=BG)
        title_box.pack(side=tk.LEFT)
        tk.Label(title_box, text="eeggamev1.0", bg=BG, fg=ACCENT,
                 font=(FONT_FAMILY, 18, "bold")).pack(side=tk.LEFT)
        tk.Label(title_box, text=" ·  脑电 AI 训练与推理控制",
                 bg=BG, fg=TXT2,
                 font=(FONT_FAMILY, 12)).pack(side=tk.LEFT, padx=(4, 0))

        # 右：功能模块下拉 + 状态点
        right = tk.Frame(top, bg=BG)
        right.pack(side=tk.RIGHT)

        tk.Label(right, text="功能模块", bg=BG, fg=TXT2,
                 font=(FONT_FAMILY, 11)).pack(side=tk.LEFT, padx=(0, 8))
        self.mode_var = tk.StringVar(value="AI 训练程序")
        self.mode_combo = ttk.Combobox(
            right, textvariable=self.mode_var, state="readonly", width=18,
            values=["AI 训练程序", "AI 推理检测程序"],
            font=(FONT_FAMILY, 11, "bold"))
        self.mode_combo.pack(side=tk.LEFT, ipady=4)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        # 状态指示点（圆点）
        self._dot = tk.Canvas(right, width=14, height=14, bg=BG,
                              highlightthickness=0)
        self._dot.pack(side=tk.LEFT, padx=(12, 4))
        self._dot_id = self._dot.create_oval(2, 2, 12, 12,
                                              fill=GRAY, outline="")
        self.mode_lbl = tk.Label(right, text="就绪", bg=BG, fg=TXT2,
                                  font=(FONT_FAMILY, 10, "bold"))
        self.mode_lbl.pack(side=tk.LEFT, padx=(0, 0))

        # 数据目录入口 chip + ⚙ 按钮（夹在标题与功能模块之间）
        self._build_paths_chip(top)

        # 分割线
        sep = tk.Frame(self.root, bg=PANEL_BORDER, height=1)
        sep.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(8, 0))

    def _build_paths_chip(self, parent):
        """紧凑显示当前 3 个路径摘要 + ⚙ 设置按钮。"""
        chip_frame = tk.Frame(parent, bg=BG)
        chip_frame.pack(side=tk.LEFT, padx=(24, 0))

        chip = tk.Frame(chip_frame, bg=PANEL, bd=1, relief=tk.FLAT,
                        highlightthickness=1, highlightbackground=PANEL_BORDER)
        chip.pack(side=tk.LEFT, ipadx=10, ipady=4)

        tk.Label(chip, text="📁", bg=PANEL, fg=ACCENT,
                 font=(FONT_FAMILY, 11, "bold")).pack(side=tk.LEFT,
                                                      padx=(0, 6))

        self.paths_summary_var = tk.StringVar(value="数据目录：加载中…")
        tk.Label(chip, textvariable=self.paths_summary_var, bg=PANEL,
                 fg=TXT2, font=(FONT_FAMILY, 9)).pack(side=tk.LEFT)

        gear = tk.Button(chip_frame, text="⚙", bg=ACCENT, fg="white",
                         activebackground=ACCENT_DARK,
                         activeforeground="white",
                         relief=tk.FLAT, font=(FONT_FAMILY, 11, "bold"),
                         cursor="hand2", command=self._open_paths_dialog,
                         width=3)
        gear.pack(side=tk.LEFT, padx=(6, 0), ipady=2)

    def _update_paths_chip(self):
        m = _abbreviate(self.cfg.get("move_dir", ""))
        s = _abbreviate(self.cfg.get("stay_dir", ""))
        e = _abbreviate(self.cfg.get("export_dir", ""))
        self.paths_summary_var.set(
            "move=%s · stay=%s · 导出=%s" % (m, s, e))

    # ====================== 路径配置弹窗 ====================== #
    def _open_paths_dialog(self):
        """弹出 Toplevel，编辑 3 个路径并保存到 eeg_config.json。"""
        if self.paths_dialog is not None and self.paths_dialog.winfo_exists():
            self.paths_dialog.lift()
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("数据目录配置")
        dlg.configure(bg=BG)
        dlg.geometry("520x420")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        self.paths_dialog = dlg

        # 顶部标题
        head = tk.Frame(dlg, bg=BG)
        head.pack(fill=tk.X, padx=20, pady=(16, 8))
        tk.Label(head, text="⚙  数据目录配置", bg=BG, fg=ACCENT,
                 font=(FONT_FAMILY, 13, "bold")).pack(anchor=tk.W)
        tk.Label(head, text="修改后点保存即可写入 eeg_config.json",
                 bg=BG, fg=TXT3,
                 font=(FONT_FAMILY, 9)).pack(anchor=tk.W, pady=(2, 0))

        # 3 个路径输入
        body = tk.Frame(dlg, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(8, 8))

        move_var = tk.StringVar(value=self.cfg.get("move_dir", ""))
        stay_var = tk.StringVar(value=self.cfg.get("stay_dir", ""))
        export_var = tk.StringVar(value=self.cfg.get("export_dir", ""))

        def make_row(parent, label, var, key, row):
            box = tk.Frame(parent, bg=BG)
            box.pack(fill=tk.X, pady=(0, 10))
            tk.Label(box, text=label, bg=BG, fg=TXT2,
                     font=(FONT_FAMILY, 10, "bold")).pack(anchor=tk.W,
                                                          pady=(0, 4))
            rowf = tk.Frame(box, bg=BG)
            rowf.pack(fill=tk.X)
            ent = tk.Entry(rowf, textvariable=var, bg=ENTRY_BG, fg=TXT,
                           relief=tk.FLAT, bd=1, highlightthickness=1,
                           highlightbackground=ENTRY_BORDER,
                           highlightcolor=ACCENT,
                           font=(FONT_FAMILY, 9))
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

            def browse():
                d = filedialog.askdirectory(
                    initialdir=var.get() or BASE_DIR, title="选择 %s" % label,
                    parent=dlg)
                if d:
                    var.set(d)
            btn = tk.Button(rowf, text="浏览", bg=BUTTON_BG, fg=TXT,
                            activebackground=PANEL, relief=tk.FLAT, bd=1,
                            highlightthickness=1,
                            highlightbackground=ENTRY_BORDER,
                            font=(FONT_FAMILY, 9), cursor="hand2",
                            command=browse)
            btn.pack(side=tk.RIGHT, padx=(6, 0), ipady=2)
            return ent

        make_row(body, "运动数据目录 (move)", move_var, "move", 0)
        make_row(body, "静止数据目录 (stay)", stay_var, "stay", 1)
        make_row(body, "模型导出目录", export_var, "export", 2)

        # 底部按钮区
        foot = tk.Frame(dlg, bg=PANEL, bd=1, relief=tk.FLAT,
                        highlightthickness=0)
        foot.pack(fill=tk.X, side=tk.BOTTOM)

        def on_save():
            new_cfg = {
                "move_dir": move_var.get().strip(),
                "stay_dir": stay_var.get().strip(),
                "export_dir": export_var.get().strip(),
            }
            try:
                config_store.save_config(new_cfg)
            except Exception as e:
                messagebox.showerror("保存失败", str(e), parent=dlg)
                return
            self.cfg = new_cfg
            self._update_paths_chip()
            self._broadcast_paths()
            self._on_status("已保存到 eeg_config.json", "ok")
            dlg.destroy()
            self.paths_dialog = None

        def on_cancel():
            dlg.destroy()
            self.paths_dialog = None

        save_btn = tk.Button(foot, text="保存", bg=ACCENT, fg="white",
                             activebackground=ACCENT_DARK,
                             activeforeground="white",
                             relief=tk.FLAT, font=(FONT_FAMILY, 10, "bold"),
                             cursor="hand2", command=on_save)
        save_btn.pack(side=tk.RIGHT, padx=(0, 12), pady=8, ipady=4, ipadx=18)

        cancel_btn = tk.Button(foot, text="取消", bg=BUTTON_BG, fg=TXT,
                               activebackground=PANEL, relief=tk.FLAT,
                               bd=1, highlightthickness=1,
                               highlightbackground=ENTRY_BORDER,
                               font=(FONT_FAMILY, 10),
                               cursor="hand2", command=on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 6), pady=8, ipady=4, ipadx=12)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

    def _broadcast_paths(self):
        """把当前 cfg 推到两个面板。"""
        if self.train_panel is not None:
            self.train_panel.set_paths(self.cfg)
        if self.infer_panel is not None:
            self.infer_panel.set_paths(self.cfg)

    def _on_infer_paths_changed(self):
        """推理面板保存配置后回调：从磁盘重新加载 + 刷新 chip + 广播。"""
        try:
            self.cfg = config_store.load_config()
        except Exception:
            return
        self._update_paths_chip()
        self._broadcast_paths()

    # ====================== 中段：面板容器 ====================== #
    def _build_body(self):
        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)

    def _create_panels(self):
        self.train_panel = TrainPanel(self.body,
                                       paths_dict=self.cfg,
                                       on_status=self._on_status)
        self.infer_panel = InferPanel(self.body,
                                       paths_dict=self.cfg,
                                       on_status=self._on_status,
                                       on_paths_changed=self._on_infer_paths_changed)

    def _switch_panel(self, which):
        for p, name in [(self.train_panel, "train"),
                         (self.infer_panel, "infer")]:
            if name == which:
                p.pack(fill=tk.BOTH, expand=True)
            else:
                p.pack_forget()
        if which == "train":
            self._dot.itemconfig(self._dot_id, fill=ACCENT)
        else:
            self._dot.itemconfig(self._dot_id, fill=GREEN)

    def _on_mode_change(self, event=None):
        sel = self.mode_var.get()
        if sel == "AI 训练程序":
            self._switch_panel("train")
        elif sel == "AI 推理检测程序":
            self._switch_panel("infer")

    # ====================== 状态栏 ====================== #
    def _build_statusbar(self):
        sep = tk.Frame(self.root, bg=PANEL_BORDER, height=1)
        sep.pack(side=tk.BOTTOM, fill=tk.X)

        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="就绪 · v1.0")
        tk.Label(bar, textvariable=self.status_var, bg=PANEL, fg=TXT2,
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT, padx=16, pady=6)

        tk.Label(bar, text="配置：%s" % os.path.basename(config_store.CONFIG_PATH),
                 bg=PANEL, fg=TXT3,
                 font=(FONT_FAMILY, 9)).pack(side=tk.RIGHT, padx=16, pady=6)

    def _on_status(self, text, level="info"):
        color_map = {"info": TXT2, "ok": GREEN, "warn": AMBER, "err": RED}
        self.status_var.set(text)
        self.mode_lbl.config(text={"info": "就绪", "ok": "运行",
                                    "warn": "提示", "err": "错误"}.get(level, "就绪"),
                              fg=color_map.get(level, TXT2))
        try:
            if level == "ok":
                self._dot.itemconfig(self._dot_id, fill=GREEN)
            elif level in ("warn", "err"):
                self._dot.itemconfig(self._dot_id,
                                       fill=AMBER if level == "warn" else RED)
        except Exception:
            pass

    # ====================== 退出钩子 ====================== #
    def on_close(self):
        try:
            if self.infer_panel:
                self.infer_panel.cleanup()
        except Exception:
            pass
        try:
            if self.train_panel and self.train_panel.training:
                self.train_panel.stop_event.set()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    root = tk.Tk()

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TCombobox", fieldbackground=ENTRY_BG,
                    background=ENTRY_BG, foreground=TXT,
                    selectbackground=ACCENT, selectforeground="white")
    style.map("TCombobox",
              fieldbackground=[("readonly", ENTRY_BG)],
              foreground=[("readonly", TXT)])
    style.configure("EEG.Horizontal.TProgressbar",
                    troughcolor="#F3F4F6", background=ACCENT, thickness=14)

    app = MainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
