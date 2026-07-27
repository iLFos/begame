# -*- coding: utf-8 -*-
"""
基于 be_stay_a_move_data (move=运动 / stay=静止) 训练 EEGCNN，
导出与 eeg_inference_gui.py 完全一致的权重字典 (eeg_cnn_best_weights.npy)
与标准化参数 (scaler_mean.npy / scaler_scale.npy)。
纯 NumPy 实现（含向量化 conv / batchnorm / maxpool 反向传播），无需 PyTorch。
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = r"D:\wangLeo\wangleo\2026yc\be_stay_a_move_data"
WINDOW_SIZE = 128
STRIDE = 2
CHANNELS = 12
SEED = 42
np.random.seed(SEED)


# ---------------- 数据 ----------------
def load_dataset(src_dir, ws=WINDOW_SIZE, stride=STRIDE, chs=CHANNELS):
    X, y = [], []
    for label, sub in [(1, "move"), (0, "stay")]:
        d = os.path.join(src_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(d, fn))
            sig = df.select_dtypes(include=[np.number]).values.astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0)[:, :chs]
            n = sig.shape[0]
            if n < ws:
                continue
            s = 0
            while s + ws <= n:
                X.append(sig[s:s + ws])
                y.append(label)
                s += stride
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------- 标准化 (复刻 sklearn StandardScaler, ddof=0) ----------------
class Scaler:
    def fit(self, X):
        Xf = X.reshape(X.shape[0], -1).astype(np.float64)
        self.mean = Xf.mean(0)
        self.scale = Xf.std(0)
        return self

    def transform(self, X):
        Xf = X.reshape(X.shape[0], -1).astype(np.float64)
        out = (Xf - self.mean) / self.scale
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -5, 5).reshape(X.shape).astype(np.float32)


# ---------------- 前向算子 ----------------
def im2col(x, K):
    N, Cin, L = x.shape
    Lout = L - K + 1
    cols = np.lib.stride_tricks.as_strided(
        x, shape=(N, Cin, Lout, K),
        strides=(x.strides[0], x.strides[1], x.strides[2], x.strides[2]))
    return cols  # (N, Cin, Lout, K)


def conv_forward(x, W, b):
    N, Cin, L = x.shape
    Cout, _, K = W.shape
    Lout = L - K + 1
    cols = im2col(x, K).reshape(N * Lout, Cin * K)
    Wf = W.reshape(Cout, Cin * K)
    out = cols @ Wf.T
    out = out.reshape(N, Lout, Cout).transpose(0, 2, 1)
    return out + b, cols


def conv_backward(dout, cols, W, in_shape):
    N, Cin, L = in_shape
    Cout, _, K = W.shape
    Lout = L - K + 1
    dout_flat = dout.transpose(0, 2, 1).reshape(N * Lout, Cout)
    cols_flat = cols.reshape(N * Lout, Cin * K)
    dWf = dout_flat.T @ cols_flat
    dW = dWf.reshape(Cout, Cin, K)
    db = dout_flat.sum(0).reshape(Cout, 1)
    dcols_flat = dout_flat @ W.reshape(Cout, Cin * K)
    dcols = dcols_flat.reshape(N, Cin, Lout, K)
    dx = np.zeros((N, Cin, L), dtype=np.float32)
    for k in range(K):
        dx[:, :, k:k + Lout] += dcols[:, :, :, k]
    return dW, db, dx


def maxpool_forward(x, p=2):
    N, C, L = x.shape
    Lo = L // p
    return x[:, :, :Lo * p].reshape(N, C, Lo, p).max(axis=-1)


def maxpool_backward(dp, a, p=2):
    N, C, L = a.shape
    Lo = L // p
    da = np.zeros_like(a)
    for j in range(Lo):
        m = a[:, :, 2 * j] >= a[:, :, 2 * j + 1]
        da[:, :, 2 * j] = dp[:, :, j] * m
        da[:, :, 2 * j + 1] = dp[:, :, j] * (~m)
    return da


def bn_forward(x, gamma, beta, rm, rv, momentum=0.9, training=True):
    N, C, L = x.shape
    mu = x.mean((0, 2))
    var = x.var((0, 2))
    std = np.sqrt(var + 1e-5)
    xhat = (x - mu[None, :, None]) / std[None, :, None]
    if training:
        rm[:] = momentum * rm + (1 - momentum) * mu
        rv[:] = momentum * rv + (1 - momentum) * var
    out = gamma[None, :, None] * xhat + beta[None, :, None]
    return out, (xhat, gamma, std)


def bn_backward(dout, cache):
    xhat, gamma, std = cache
    N, C, L = dout.shape
    Ntot = N * L
    dgamma = (dout * xhat).sum((0, 2))
    dbeta = dout.sum((0, 2))
    dxhat = dout * gamma[None, :, None]
    dx = (1.0 / (Ntot * std[None, :, None])) * (
        Ntot * dxhat - dxhat.sum((0, 2), keepdims=True)
        - xhat * (dxhat * xhat).sum((0, 2), keepdims=True))
    return dx, dgamma, dbeta


def softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ---------------- 参数初始化 ----------------
def init_params():
    p = {}
    p["W1"] = (np.random.randn(4, 12, 5) * 0.1).astype(np.float32)
    p["b1"] = np.zeros((4, 1), dtype=np.float32)
    p["W2"] = (np.random.randn(8, 4, 3) * 0.1).astype(np.float32)
    p["b2"] = np.zeros((8, 1), dtype=np.float32)
    p["W3"] = (np.random.randn(8, 2) * 0.1).astype(np.float32)
    p["b3"] = np.zeros((1, 2), dtype=np.float32)
    p["gamma1"] = np.ones(4, dtype=np.float32)
    p["beta1"] = np.zeros(4, dtype=np.float32)
    p["gamma2"] = np.ones(8, dtype=np.float32)
    p["beta2"] = np.zeros(8, dtype=np.float32)
    p["running_mean1"] = np.zeros(4, dtype=np.float32)
    p["running_var1"] = np.ones(4, dtype=np.float32)
    p["running_mean2"] = np.zeros(8, dtype=np.float32)
    p["running_var2"] = np.ones(8, dtype=np.float32)
    return p


def forward(p, x, training=True):
    c = {}
    z1, cols1 = conv_forward(x, p["W1"], p["b1"]); c["cols1"] = cols1
    b1o, c["bn1"] = bn_forward(z1, p["gamma1"], p["beta1"],
                               p["running_mean1"], p["running_var1"], training=training)
    a1 = np.maximum(0, b1o); c["a1"] = a1
    p1 = maxpool_forward(a1, 2); c["p1"] = p1
    z2, cols2 = conv_forward(p1, p["W2"], p["b2"]); c["cols2"] = cols2
    b2o, c["bn2"] = bn_forward(z2, p["gamma2"], p["beta2"],
                               p["running_mean2"], p["running_var2"], training=training)
    a2 = np.maximum(0, b2o); c["a2"] = a2
    p2 = maxpool_forward(a2, 2)
    g = p2.mean(-1); c["g"] = g
    logits = g @ p["W3"] + p["b3"]
    return logits, c


def train():
    X, y = load_dataset(SRC_DIR)
    X = np.transpose(X, (0, 2, 1))  # (N, C=12, T=128) 与 GUI 约定一致
    print(f"数据集: 样本={X.shape[0]}  运动={int((y == 1).sum())}  静止={int((y == 0).sum())}")
    scaler = Scaler().fit(X)
    Xn = scaler.transform(X)

    p = init_params()
    lr = 0.3
    epochs = 500
    N = Xn.shape[0]
    eye = np.eye(2)

    for ep in range(epochs):
        logits, c = forward(p, Xn, training=True)
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-8))

        dlogits = (probs - eye[y]) / N
        g = c["g"]
        dW3 = g.T @ dlogits
        db3 = dlogits.sum(0, keepdims=True)
        dg = dlogits @ p["W3"].T
        L2 = int(c["a2"].shape[2] // 2)  # p2 时间维
        dp2 = (dg / L2)[:, :, None].repeat(L2, axis=2)
        da2 = maxpool_backward(dp2, c["a2"], 2)
        dA2, dgamma2, dbeta2 = bn_backward(da2, c["bn2"])
        dW2, db2, dp1 = conv_backward(dA2, c["cols2"], p["W2"], c["p1"].shape)
        da1 = maxpool_backward(dp1, c["a1"], 2)
        dA1, dgamma1, dbeta1 = bn_backward(da1, c["bn1"])
        dW1, db1, _ = conv_backward(dA1, c["cols1"], p["W1"], (N, 12, 128))

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "W3": dW3, "b3": db3, "gamma1": dgamma1, "beta1": dbeta1,
                 "gamma2": dgamma2, "beta2": dbeta2}
        for k, d in grads.items():
            p[k] = (p[k] - lr * np.clip(d, -100, 100).astype(np.float32)).astype(np.float32)

        if ep % 10 == 0 or ep == epochs - 1:
            pred = probs.argmax(1)
            print(f"epoch {ep:3d}  loss={loss:.4f}  acc={(pred == y).mean():.3f}")

    logits, _ = forward(p, Xn, training=False)
    pred = softmax(logits).argmax(1)
    print(f"训练集准确率: {(pred == y).mean():.3f}")

    weights = {k: np.asarray(v) for k, v in p.items()}
    np.save(os.path.join(BASE_DIR, "eeg_cnn_best_weights.npy"), weights)
    np.save(os.path.join(BASE_DIR, "scaler_mean.npy"), scaler.mean)
    np.save(os.path.join(BASE_DIR, "scaler_scale.npy"), scaler.scale)
    print("已保存权重与标准化参数到", BASE_DIR)


if __name__ == "__main__":
    train()
