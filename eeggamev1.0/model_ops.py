# -*- coding: utf-8 -*-
"""
model_ops.py

从 EEG_Football_Realtime/train_model.py 移植的纯 NumPy 算子层：
  - im2col / conv_forward / conv_backward
  - maxpool_forward / maxpool_backward
  - bn_forward / bn_backward （批归一化）
  - softmax
  - Scaler （复刻 sklearn StandardScaler, ddof=0）

本文件只提供底层算子，不含任何可配置的网络结构，供 model.py 引用，
从而保证与实时推理端 (realtime_core.py) 的前向数值完全一致。
"""
import numpy as np


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

    def save(self, mean_path, scale_path):
        np.save(mean_path, np.asarray(self.mean))
        np.save(scale_path, np.asarray(self.scale))


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