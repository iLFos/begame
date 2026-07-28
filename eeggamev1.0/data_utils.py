# -*- coding: utf-8 -*-
"""
data_utils.py

从 move / stay 两个目录加载脑电 CSV，滑窗构建训练样本。

CSV 格式（13 列）：
  timestamp, signalQuality, attention, meditation, delta, theta,
  lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, midGamma, label
其中 label 列恒为空；实际使用的 12 个数值通道 = signalQuality..midGamma。

约定：move 目录 -> 标签 1（运动），stay 目录 -> 标签 0（静止）。
"""
import os
import numpy as np
import pandas as pd

from signal_filter import preprocess

CHANNELS = 12
DEFAULT_WINDOW = 128
DEFAULT_STRIDE = 2

# signalQuality 通道索引（不应被带通/陷波影响）
SIGNAL_QUALITY_CH = 0


def load_one_csv(path, channels=CHANNELS, preprocess_opts=None,
                 fs=128.0):
    """读取单个 CSV，返回清洗后的 (n, channels) 数值数组。

    参数：
      path             - CSV 路径
      channels         - 输出通道数（默认 12）
      preprocess_opts  - 预处理选项 dict，None 或空表示不处理；
                         可通过 skip_channels 指定跳过某些通道（默认跳过 signalQuality）。
      fs               - 采样率（默认 128 Hz，与窗口 128 = 1 秒对应）

    说明：原始 CSV 含 13 列（timestamp + 11 个 TGAM 数值特征 + 空 label）。
    pandas 读出的数值列只有 11 个（signalQuality..midGamma），label 列为空。
    为与实时推理端 (realtime_core，输入固定 12 通道、第 12 通道为全零 label)
    保持兼容，这里将数值列补全到 12 通道，第 12 通道补 0。
    """
    df = pd.read_csv(path)
    sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0)
    if sig.shape[1] > channels:
        sig = sig[:, :channels]
    elif sig.shape[1] < channels:
        pad = np.zeros((sig.shape[0], channels - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)

    # 预处理（仅作用于启用项）
    if preprocess_opts:
        skip = set(preprocess_opts.get("skip_channels", [SIGNAL_QUALITY_CH]))
        # 拆分：需预处理的列 / 不动的列
        all_ch = list(range(sig.shape[1]))
        keep_idx = [i for i in all_ch if i in skip]
        proc_idx = [i for i in all_ch if i not in skip]
        if proc_idx:
            sig_proc = sig[:, proc_idx]
            sig_proc = preprocess(sig_proc, fs, preprocess_opts)
            sig[:, proc_idx] = sig_proc

    return sig


def load_dataset(move_dir, stay_dir, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE,
                 channels=CHANNELS, val_ratio=0.0, seed=42,
                 preprocess_opts=None, fs=128.0):
    """从两个目录构建滑窗样本。

    参数：
      move_dir / stay_dir - 数据目录
      window / stride     - 滑窗大小与步长
      channels            - 通道数
      val_ratio           - 验证集比例（0 表示不切分）
      preprocess_opts     - 预处理选项 dict，None 表示不处理
      fs                  - 采样率（用于滤波）

    返回：
      (X, y)                     若 val_ratio==0
      (X_tr, y_tr, X_val, y_val) 若 val_ratio>0
    其中 X 形状 (N, channels, window) 与模型输入约定一致。
    """
    X, y = [], []
    for label, d in [(1, move_dir), (0, stay_dir)]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".csv"):
                continue
            sig = load_one_csv(os.path.join(d, fn), channels=channels,
                               preprocess_opts=preprocess_opts, fs=fs)
            n = sig.shape[0]
            if n < window:
                continue
            s = 0
            while s + window <= n:
                X.append(sig[s:s + window])
                y.append(label)
                s += stride

    if len(X) == 0:
        raise RuntimeError("未找到足够长度的脑电样本，请检查 move/stay 目录与窗口大小。")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # 调整为 (N, C, T)
    X = np.transpose(X, (0, 2, 1))

    if val_ratio and 0.0 < val_ratio < 1.0:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
        n_val = int(len(X) * val_ratio)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        return (X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])

    return X, y


def dataset_summary(move_dir, stay_dir, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE,
                    channels=CHANNELS):
    """返回数据集概览信息（不构建样本，仅统计）。"""
    info = {"move_files": [], "stay_files": [], "move_rows": 0, "stay_rows": 0,
            "move_range": None, "stay_range": None}
    for key, d in [("move", move_dir), ("stay", stay_dir)]:
        if not os.path.isdir(d):
            continue
        vmin, vmax = np.inf, -np.inf
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".csv"):
                continue
            try:
                sig = load_one_csv(os.path.join(d, fn), channels=channels)
            except Exception:
                continue
            n = sig.shape[0]
            wins = max(0, (n - window) // stride + 1) if n >= window else 0
            info[key + "_files"].append((fn, n, wins))
            info[key + "_rows"] += n
            if sig.size:
                vmin = float(min(vmin, sig.min()))
                vmax = float(max(vmax, sig.max()))
        info[key + "_range"] = (vmin, vmax) if np.isfinite(vmin) else None
    return info


def csv_summary(path, channels=CHANNELS, max_rows=200000):
    """返回单个 CSV 的统计信息（用于预览窗口）。

    返回：
      {
        "path": str,
        "rows": int,
        "columns": list[str],
        "per_channel": [(name, min, max, mean, std, nan_count), ...],
        "quality_mean": float,
        "duration_s": float,
      }
    """
    df = pd.read_csv(path)
    n_rows = len(df)
    use_df = df.iloc[:max_rows] if n_rows > max_rows else df
    num = use_df.select_dtypes(include=[np.number])
    names = [str(c) for c in num.columns][:channels]
    arr = num.values.astype(np.float32)
    if arr.shape[1] > channels:
        arr = arr[:, :channels]
    elif arr.shape[1] < channels:
        pad = np.zeros((arr.shape[0], channels - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    per_ch = []
    for c in range(arr.shape[1]):
        col = arr[:, c]
        nan_n = int(np.isnan(col).sum())
        col = np.nan_to_num(col, nan=0.0)
        per_ch.append((names[c] if c < len(names) else f"ch{c}",
                       float(col.min()), float(col.max()),
                       float(col.mean()), float(col.std()), nan_n))
    qcol = arr[:, 0] if arr.shape[1] > 0 else np.array([])
    return {
        "path": path,
        "rows": n_rows,
        "columns": names,
        "per_channel": per_ch,
        "quality_mean": float(qcol.mean()) if qcol.size else 0.0,
        "duration_s": n_rows / 128.0,  # 假设 128 Hz
    }


def export_dataset_csv(move_dir, stay_dir, out_path, channels=CHANNELS,
                       preprocess_opts=None, fs=128.0,
                       append_label=True):
    """把 move+stay 数据集合并导出为单个 CSV。

    每行 = 一个采样点；可选在末尾追加 label 列（move=1, stay=0）。
    适用于「从多个原始 CSV 合并出一个干净的数据集」。

    返回：(n_move, n_stay, total)
    """
    import csv as _csv
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cols = ["timestamp", "signalQuality", "attention", "meditation",
            "delta", "theta", "lowAlpha", "highAlpha", "lowBeta",
            "highBeta", "lowGamma", "midGamma"]
    if append_label:
        cols = cols + ["label"]

    n_move = n_stay = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for label, d in [(1, move_dir), (0, stay_dir)]:
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(".csv"):
                    continue
                sig = load_one_csv(os.path.join(d, fn),
                                   channels=channels,
                                   preprocess_opts=preprocess_opts,
                                   fs=fs)
                # 简化：timestamp 重新按行号递增
                base = (1 if label == 1 else 1000000)
                for i in range(sig.shape[0]):
                    row = [base + i] + sig[i].tolist()
                    if append_label:
                        row.append(label)
                    w.writerow(row)
                    if label == 1:
                        n_move += 1
                    else:
                        n_stay += 1
    return n_move, n_stay, n_move + n_stay
