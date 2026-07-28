# -*- coding: utf-8 -*-
"""
realtime_core.py

实时滑窗推理核心（纯 NumPy，复用 model.forward）。

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

阈值机制：threshold_value ∈ [0, 1]。仅当 prob[1] ≥ threshold_value 时，才向
movda.txt 写入 "1"（红方前进）；否则写 "0"（停止）。threshold_value 由 UI 滑块
实时设置（slider ∈ [1, 100] → threshold_value = slider / 100）。
"""

import os
import threading
import numpy as np

import model as _model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 128
CHANNELS = 12
REINFER_EVERY = 16
MOVDA_PATH = os.path.join(BASE_DIR, "movda.txt")


class RealtimeInference:
    """实时推理核心。线程安全，持有滑窗环形缓冲与最新特征快照。"""

    def __init__(self, model_path=None, on_result=None, on_log=None,
                 threshold=0.5):
        self.model_path = model_path or os.path.join(BASE_DIR, "eeg_cnn_best_weights.npy")
        self.on_result = on_result
        self.on_log = on_log
        self.model = None
        self.mean = None
        self.scale = None
        self._lock = threading.Lock()
        self._loaded = False

        # 阈值：prob[1] ≥ threshold → 判定为运动（写 1）
        self.threshold = float(threshold)

        # RAW 环形缓冲（RAW 模式）
        self.buf = []
        self._since_infer = 0
        # 特征模式：最近一次 12 通道特征快照
        self.latest_feat = None

        # 当前已写入 movda.txt 的最新值，避免无意义重复写
        self._last_written = None

    # ------------------------------------------------------------------ #
    def load_model(self):
        w = np.load(self.model_path, allow_pickle=True).item()
        self._weights = w
        mp, sp = self._find_scaler()
        self.mean = np.load(mp).reshape(CHANNELS, WINDOW_SIZE).astype(np.float32)
        self.scale = np.load(sp).reshape(CHANNELS, WINDOW_SIZE).astype(np.float32)
        # 防御：训练数据中某些位置方差为 0 -> scale=0，除以 0 会产生 NaN。
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        self._loaded = True
        if self.on_log:
            self.on_log("模型已加载: %s" % os.path.basename(self.model_path), "ok")

    def _find_scaler(self):
        base = os.path.splitext(self.model_path)[0]
        cand = [base + "_scaler_mean.npy", os.path.join(BASE_DIR, "scaler_mean.npy")]
        mean = next((p for p in cand if os.path.exists(p)), cand[-1])
        cand2 = [base + "_scaler_scale.npy", os.path.join(BASE_DIR, "scaler_scale.npy")]
        scale = next((p for p in cand if os.path.exists(p)), cand[-1])
        return mean, scale

    # ------------------------------------------------------------------ #
    def set_threshold(self, value):
        """实时更新阈值（value ∈ [0, 1]）。"""
        try:
            self.threshold = float(value)
        except Exception:
            pass

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
        window = np.array(self.buf[-WINDOW_SIZE:], dtype=np.float32)
        win12 = np.repeat(window[np.newaxis, :], CHANNELS, axis=0)
        self._predict_and_write(win12)

    # ------------------------------------------------------------------ #
    # 模式 3：离线回放（直接喂 (时间,通道) 窗口，与训练一致）
    def infer_window(self, window_tc):
        """window_tc: (WINDOW_SIZE, CHANNELS) 或 (>=128, CHANNELS) 的数组。"""
        arr = np.asarray(window_tc, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.repeat(arr[:, None], CHANNELS, axis=1)
        else:
            if arr.shape[0] < WINDOW_SIZE:
                pad = WINDOW_SIZE - arr.shape[0]
                arr = np.concatenate([np.zeros((pad, arr.shape[1]), dtype=np.float32), arr], axis=0)
            arr = arr[-WINDOW_SIZE:]
        win12 = arr.T.astype(np.float32)
        return self._predict_and_write(win12, write=True)

    # ------------------------------------------------------------------ #
    def _predict_and_write(self, win12, write=True):
        """win12: (12,128)。标准化 -> 前向 -> softmax -> 阈值判定 -> 写文件。"""
        norm = np.clip((win12 - self.mean) / self.scale, -5, 5).astype(np.float32)
        xin = norm[np.newaxis, :, :]
        logits, _ = _model.forward(self._weights, xin, training=False)
        e = np.exp(logits - logits.max(1, keepdims=True))
        prob = (e / e.sum(1, keepdims=True))[0]
        pred = int(np.argmax(prob))
        prob_move = float(prob[1])

        # 阈值判定：仅当运动概率 ≥ 阈值时才写 "1"
        if write:
            final_signal = 1 if prob_move >= self.threshold else 0
            self._write_signal(final_signal)

        if self.on_result:
            self.on_result(pred, prob_move, prob.tolist(), win12)
        return pred, prob_move

    # ------------------------------------------------------------------ #
    def _write_signal(self, value):
        if self._last_written == value:
            return
        try:
            with open(MOVDA_PATH, "w", encoding="utf-8") as f:
                f.write(str(int(value)))
            self._last_written = int(value)
        except Exception as e:
            if self.on_log:
                self.on_log("写 movda.txt 失败: %s" % e, "err")


# ---------------------------------------------------------------------- #
def replay_csv(csv_path, inferencer, stride=2, on_window=None):
    """离线 CSV 回放：把 CSV 前 12 个数值列按时间滑窗喂入，逐窗推理。"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0)
    if sig.shape[1] > CHANNELS:
        sig = sig[:, :CHANNELS]
    elif sig.shape[1] < CHANNELS:
        pad = np.zeros((sig.shape[0], CHANNELS - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)
    n = sig.shape[0]
    preds = []
    for s in range(0, n - WINDOW_SIZE + 1, stride):
        w = sig[s:s + WINDOW_SIZE]
        pred, pm = inferencer.infer_window(w)
        preds.append(pred)
        if on_window:
            on_window(pred, pm, w)
    return np.array(preds)