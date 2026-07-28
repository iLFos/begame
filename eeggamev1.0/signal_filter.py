# -*- coding: utf-8 -*-
"""
signal_filter.py

训练端的纯 numpy 信号预处理模块，用于「过滤杂波」：

  - detrend(x)                          去除基线（按通道减均值）
  - iir_lowpass / iir_highpass          二阶 Butterworth（RBJ biquad）
  - iir_bandpass(fs, low, high)         二阶 Butterworth 带通
  - iir_notch(x, fs, freq, Q=30)        窄带陷波，去工频（50/60 Hz）
  - clip_outliers(x, n_sigma=3.5)       按通道 n×σ 截断离群点
  - preprocess(sig, fs, opts)           总入口，根据 opts dict 依次调用

约定：
  - 输入 sig 形状 (T, C)，T = 时间步，C = 通道。
  - 输出与输入同形状 dtype float32。
  - 滤波器系数采用 RBJ Audio EQ Cookbook 标准二阶 biquad，
    直接 II 形式实现（不引入 scipy）。

参考：
  https://www.musicdsp.org/en/latest/Filters/197-rbj-audio-eq-cookbook.html
"""
import numpy as np


# ------------------------------------------------------------------ #
# 基础工具
# ------------------------------------------------------------------ #
def _to_2d(sig):
    arr = np.asarray(sig)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def detrend(x):
    """按通道减去均值（去基线）。"""
    a = _to_2d(x)
    return (a - a.mean(axis=0, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------------ #
# IIR 直接 II 滤波（双 biquad filtfilt → 零相位）
# ------------------------------------------------------------------ #
def _lfilter(b, a, x):
    """线性 IIR/FIR 滤波（直接 II 形式）。x 一维。"""
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    a0 = a[0]
    if a0 == 0:
        raise ZeroDivisionError("a[0] 不能为 0")
    b = b / a0
    a = a / a0
    n = max(len(a), len(b))
    b = np.concatenate([b, np.zeros(n - len(b))])
    a = np.concatenate([a, np.zeros(n - len(a))])
    y = np.empty_like(x, dtype=np.float64)
    nb = len(b)
    na = len(a)
    for i in range(len(x)):
        acc = 0.0
        kmax = min(i + 1, nb)
        for k in range(kmax):
            acc += b[k] * x[i - k]
        kmax2 = min(i + 1, na - 1)
        for k in range(1, kmax2 + 1):
            acc -= a[k] * y[i - k]
        y[i] = acc
    return y


def _filtfilt(b, a, x):
    """零相位滤波（前向 + 反向）。x 形状 (T,) 或 (T, C)。"""
    x = np.asarray(x, dtype=np.float64)
    squeeze = (x.ndim == 1)
    if squeeze:
        x = x[:, None]
    T, C = x.shape
    out = np.empty_like(x)
    for c in range(C):
        xs = x[:, c]
        y1 = _lfilter(b, a, xs)
        y2 = _lfilter(b, a, y1[::-1])[::-1]
        out[:, c] = y2
    if squeeze:
        out = out[:, 0]
    # 数值保护：清理 NaN/Inf，并限制在 float32 合理范围内
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    out = np.clip(out, -1e6, 1e6)
    return out.astype(np.float32)


# ------------------------------------------------------------------ #
# Butterworth IIR（RBJ 二阶 biquad）
# ------------------------------------------------------------------ #
def _biquad_lowpass(fs, fc, Q=0.7071):
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_highpass(fs, fc, Q=0.7071):
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_bandpass(fs, f_low, f_high):
    """Constant skirt gain bandpass，Q 由带宽决定。"""
    f0 = np.sqrt(f_low * f_high)
    Q = f0 / max(f_high - f_low, 1e-9)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def iir_lowpass(x, fs, fc, Q=0.7071):
    a2 = _to_2d(x)
    b, a = _biquad_lowpass(fs, fc, Q)
    return _filtfilt(b, a, a2)


def iir_highpass(x, fs, fc, Q=0.7071):
    a2 = _to_2d(x)
    b, a = _biquad_highpass(fs, fc, Q)
    return _filtfilt(b, a, a2)


def iir_bandpass(x, fs, f_low, f_high):
    a2 = _to_2d(x)
    b, a = _biquad_bandpass(fs, f_low, f_high)
    return _filtfilt(b, a, a2)


# ------------------------------------------------------------------ #
# 陷波（窄带 IIR）+ 异常值裁剪
# ------------------------------------------------------------------ #
def iir_notch(x, fs, freq, Q=30.0):
    """RBJ 陷波器：b=[1, -2cos(w0), 1]; a=[1+alpha, -2cos(w0), 1-alpha]"""
    a2 = _to_2d(x)
    fs = float(fs); freq = float(freq); Q = float(Q)
    if not (0 < freq < fs / 2.0):
        raise ValueError("陷波频率需在 (0, fs/2) 之间")
    w0 = 2.0 * np.pi * freq / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b = np.array([1.0, -2.0 * np.cos(w0), 1.0])
    a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha])
    return _filtfilt(b, a, a2)


def clip_outliers(x, n_sigma=3.5):
    a = _to_2d(x)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True) + 1e-8
    hi = mu + n_sigma * sd
    lo = mu - n_sigma * sd
    return np.clip(a, lo, hi).astype(np.float32)


# ------------------------------------------------------------------ #
# 统一入口
# ------------------------------------------------------------------ #
def preprocess(sig, fs, opts):
    """根据 opts 字典对 sig=(T,C) 依次施加预处理。

    opts 字段（全部可选）：
        detrend      bool   是否去基线
        bandpass     bool   是否带通
        low, high    float  带通截止（Hz）
        notch        bool   是否陷波
        notch_freq   float  陷波频率（Hz）
        notch_Q      float  陷波 Q 值
        clip         bool   是否异常值裁剪
        clip_sigma   float  裁剪倍数

    返回与输入同形状的 float32 数组。
    """
    if not opts:
        return _to_2d(sig)
    a = _to_2d(sig).copy()
    fs = float(fs)

    if opts.get("detrend"):
        a = a - a.mean(axis=0, keepdims=True)
    if opts.get("bandpass"):
        a = iir_bandpass(a, fs,
                         float(opts.get("low", 1.0)),
                         float(opts.get("high", 40.0)))
    if opts.get("notch"):
        a = iir_notch(a, fs,
                      float(opts.get("notch_freq", 50.0)),
                      Q=float(opts.get("notch_Q", 30.0)))
    if opts.get("clip"):
        a = clip_outliers(a, n_sigma=float(opts.get("clip_sigma", 3.5)))
    return a.astype(np.float32)
