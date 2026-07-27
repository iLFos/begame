# -*- coding: utf-8 -*-
"""
实时滑窗推理核心（纯 NumPy，复用 train_model.forward）

关键事实（已核对）：原训练 CSV 的 12 个数值列是 TGAM 大包解析出的特征：
    signalQuality, attention, meditation, delta, theta, lowAlpha, highAlpha,
    lowBeta, highBeta, lowGamma, midGamma, label
其中 label 列为空（训练时按 0 处理）——即模型实际以 11 个 TGAM 特征通道 + 1 个
全零通道(第 12 通道) 作为输入。因此实时端应把 TGAM 大包解析出的特征喂给模型，
而不是把单通道 RAW 广播成 12 通道。

本模块支持三种输入模式：
  1) 特征模式（实时首选）：每次收到 TGAM 大包，构造 12 通道特征快照
     [signalQuality, attention, meditation, delta..midGamma, 0]，沿时间复制成
     (12,128) 窗口后推理。
  2) RAW 模式（仅 RAW 的硬件）：单通道广播到 12 通道（演示用，分类精度不保证）。
  3) 离线回放：直接把 CSV 的前 12 个数值列当作 (时间,通道) 窗口喂入（与训练一致）。

推理结果 (0/1) 实时覆盖写入 movda.txt（UTF-8 单字符），驱动 server.js -> 红方前进。
"""

import os
import threading
import numpy as np

import train_model  # 复用已验证的前向与标准化

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 128          # 时间窗长度（与训练一致）
CHANNELS = 12              # 模型输入通道数
REINFER_EVERY = 16         # RAW 模式：每累积多少新样本重推理（~30Hz @512Hz）
MOVDA_PATH = os.path.join(BASE_DIR, "movda.txt")


class RealtimeInference:
    def __init__(self, model_path=None, on_result=None, on_log=None):
        self.model_path = model_path or os.path.join(BASE_DIR, "eeg_cnn_best_weights.npy")
        self.on_result = on_result
        self.on_log = on_log
        self.model = None
        self.mean = None
        self.scale = None
        self._lock = threading.Lock()
        self._loaded = False
        # RAW 环形缓冲（RAW 模式）
        self.buf = []
        self._since_infer = 0
        # 特征模式：最近一次 12 通道特征快照
        self.latest_feat = None

    # ------------------------------------------------------------------ #
    def load_model(self):
        w = np.load(self.model_path, allow_pickle=True).item()
        self._weights = w
        self._eeg = _EEGModelWrap(w)
        mp, sp = self._find_scaler()
        # scaler 形状 (1536,) = 12*128，重塑为 (通道,时间)=(12,128) 以对齐输入
        self.mean = np.load(mp).reshape(CHANNELS, WINDOW_SIZE).astype(np.float32)
        self.scale = np.load(sp).reshape(CHANNELS, WINDOW_SIZE).astype(np.float32)
        # 防御：训练数据中某些位置方差为 0 -> scale=0，除以 0 会产生 NaN。
        # 将 0 替换为 1.0（该位置不做缩放），保证实时推理数值稳定。
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        self._loaded = True
        if self.on_log:
            self.on_log("模型已加载: %s" % os.path.basename(self.model_path), "ok")

    def _find_scaler(self):
        base = os.path.splitext(self.model_path)[0]
        cand = [base + "_scaler_mean.npy", os.path.join(BASE_DIR, "scaler_mean.npy")]
        mean = next((p for p in cand if os.path.exists(p)), cand[-1])
        cand2 = [base + "_scaler_scale.npy", os.path.join(BASE_DIR, "scaler_scale.npy")]
        scale = next((p for p in cand2 if os.path.exists(p)), cand2[-1])
        return mean, scale

    # ------------------------------------------------------------------ #
    def reset(self):
        self.buf = []
        self._since_infer = 0
        self.latest_feat = None

    # ------------------------------------------------------------------ #
    # 模式 1：特征模式（实时首选，TGAM 大包）
    def add_feature_vector(self, vec12):
        """vec12: 长度 12 的浮点列表（11 个 TGAM 特征 + 第 12 通道 0）。"""
        if not self._loaded:
            return
        with self._lock:
            self.latest_feat = np.asarray(vec12, dtype=np.float32)[:CHANNELS]
            self._infer_features()

    def _infer_features(self):
        if self.latest_feat is None:
            return
        # 构造 (12,128) 窗口：沿时间复制当前特征快照
        win12 = np.repeat(self.latest_feat[:, None], WINDOW_SIZE, axis=1)  # (12,128)
        self._predict_and_write(win12)

    # ------------------------------------------------------------------ #
    # 模式 2：RAW 模式（单通道广播，演示用）
    def add_raw(self, value):
        if not self._loaded:
            return
        with self._lock:
            self.buf.append(float(value))
            if len(self.buf) > WINDOW_SIZE:
                self.buf.pop(0)
            self._since_infer += 1
            if len(self.buf) >= WINDOW_SIZE and self._since_infer >= REINFER_EVERY:
                self._since_infer = 0
                self._infer_raw()

    def _infer_raw(self):
        window = np.array(self.buf[-WINDOW_SIZE:], dtype=np.float32)  # (128,)
        # 单通道广播为 12 通道
        win12 = np.repeat(window[np.newaxis, :], CHANNELS, axis=0)     # (12,128)
        self._predict_and_write(win12)

    # ------------------------------------------------------------------ #
    # 模式 3：离线回放（直接喂 (时间,通道) 窗口，与训练一致）
    def infer_window(self, window_tc):
        """window_tc: (WINDOW_SIZE, CHANNELS) 或 (>=128, CHANNELS) 的数组。"""
        arr = np.asarray(window_tc, dtype=np.float32)
        if arr.ndim == 1:
            # 单通道 -> 广播
            arr = np.repeat(arr[:, None], CHANNELS, axis=1)
        else:
            if arr.shape[0] < WINDOW_SIZE:
                pad = WINDOW_SIZE - arr.shape[0]
                arr = np.concatenate([np.zeros((pad, arr.shape[1]), dtype=np.float32), arr], axis=0)
            arr = arr[-WINDOW_SIZE:]
        win12 = arr.T.astype(np.float32)  # (12,128)
        return self._predict_and_write(win12, write=True)

    # ------------------------------------------------------------------ #
    def _predict_and_write(self, win12, write=True):
        """win12: (12,128)。标准化 -> 前向 -> softmax -> argmax -> 写文件。"""
        norm = np.clip((win12 - self.mean) / self.scale, -5, 5).astype(np.float32)
        xin = norm[np.newaxis, :, :]  # (1,12,128)
        logits = self._eeg.forward(xin)
        prob = _softmax(logits)[0]
        pred = int(np.argmax(prob))
        prob_move = float(prob[1])
        if write:
            self._write_signal(pred)
        if self.on_result:
            self.on_result(pred, prob_move, win12)
        return pred, prob_move

    # ------------------------------------------------------------------ #
    def _write_signal(self, value):
        try:
            with open(MOVDA_PATH, "w", encoding="utf-8") as f:
                f.write(str(int(value)))
        except Exception as e:
            if self.on_log:
                self.on_log("写 movda.txt 失败: %s" % e, "err")


# ---------------------------------------------------------------------- #
def _softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class _EEGModelWrap:
    def __init__(self, weights):
        self.w = weights

    def forward(self, x):
        logits, _ = train_model.forward(self.w, x, training=False)
        return logits


# ---------------------------------------------------------------------- #
def replay_csv(csv_path, inferencer, stride=2, on_window=None):
    """离线 CSV 回放：把 CSV 前 12 个数值列按时间滑窗喂入，逐窗推理。

    与训练一致：直接取数值列前 12 列（label 列 NaN 已由标准化/训练处理为 0）。
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0)[:, :CHANNELS]
    n = sig.shape[0]
    preds = []
    for s in range(0, n - WINDOW_SIZE + 1, stride):
        w = sig[s:s + WINDOW_SIZE]  # (128,12)
        pred, pm = inferencer.infer_window(w)
        preds.append(pred)
        if on_window:
            on_window(pred, pm, w)
    return np.array(preds)
