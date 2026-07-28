# -*- coding: utf-8 -*-
"""
model.py

可配置的纯 NumPy 1D 卷积神经网络 (EEGCNN)，用于脑电「运动/静止」二分类。
算子复用 model_ops.py，前向数值与 EEG_Football_Realtime/realtime_core.py
完全一致。导出权重字典键名与 scaler 形状与现推理端严格兼容。

默认超参数与现有 train_model.py 一致，可复现原训练结果：
  f1=4, f2=8, k1=5, k2=3, lr=0.3, epochs=500
"""
import os
import numpy as np

from model_ops import (
    conv_forward, conv_backward, maxpool_forward, maxpool_backward,
    bn_forward, bn_backward, softmax, Scaler,
)

CHANNELS = 12
WINDOW_SIZE = 128


class EEGCNN:
    """纯 NumPy 1D-CNN：conv1(bn,relu,pool) -> conv2(bn,relu,pool)
       -> global avg pool -> fc(softmax)。"""

    def __init__(self, f1=4, f2=8, k1=5, k2=3, fc=8, channels=CHANNELS,
                 window=WINDOW_SIZE, seed=42):
        self.f1 = f1
        self.f2 = f2
        self.k1 = k1
        self.k2 = k2
        self.fc = fc
        self.channels = channels
        self.window = window
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.params = None
        self.scaler = None

    # ------------------------------------------------------------------ #
    def init_params(self):
        rng = self._rng
        p = {}
        p["W1"] = (rng.randn(self.f1, self.channels, self.k1) * 0.1).astype(np.float32)
        p["b1"] = np.zeros((self.f1, 1), dtype=np.float32)
        p["W2"] = (rng.randn(self.f2, self.f1, self.k2) * 0.1).astype(np.float32)
        p["b2"] = np.zeros((self.f2, 1), dtype=np.float32)
        # 与推理端 train_model.py 严格一致：W3 输入维 = f2（GAP 后特征维），无独立 fc 隐藏层
        p["W3"] = (rng.randn(self.f2, 2) * 0.1).astype(np.float32)
        p["b3"] = np.zeros((1, 2), dtype=np.float32)
        p["gamma1"] = np.ones(self.f1, dtype=np.float32)
        p["beta1"] = np.zeros(self.f1, dtype=np.float32)
        p["gamma2"] = np.ones(self.f2, dtype=np.float32)
        p["beta2"] = np.zeros(self.f2, dtype=np.float32)
        p["running_mean1"] = np.zeros(self.f1, dtype=np.float32)
        p["running_var1"] = np.ones(self.f1, dtype=np.float32)
        p["running_mean2"] = np.zeros(self.f2, dtype=np.float32)
        p["running_var2"] = np.ones(self.f2, dtype=np.float32)
        self.params = p
        return p

    # ------------------------------------------------------------------ #
    def forward(self, x, training=True):
        p = self.params
        c = {}
        z1, cols1 = conv_forward(x, p["W1"], p["b1"]); c["cols1"] = cols1
        b1o, c["bn1"] = bn_forward(z1, p["gamma1"], p["beta1"],
                                   p["running_mean1"], p["running_var1"],
                                   training=training)
        a1 = np.maximum(0, b1o); c["a1"] = a1
        p1 = maxpool_forward(a1, 2); c["p1"] = p1
        z2, cols2 = conv_forward(p1, p["W2"], p["b2"]); c["cols2"] = cols2
        b2o, c["bn2"] = bn_forward(z2, p["gamma2"], p["beta2"],
                                   p["running_mean2"], p["running_var2"],
                                   training=training)
        a2 = np.maximum(0, b2o); c["a2"] = a2
        p2 = maxpool_forward(a2, 2)
        g = p2.mean(-1); c["g"] = g
        logits = g @ p["W3"] + p["b3"]
        return logits, c

    # ------------------------------------------------------------------ #
    def _backward(self, p, c, dlogits, N):
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
        dW1, db1, _ = conv_backward(dA1, c["cols1"], p["W1"], (N, self.channels, self.window))
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "W3": dW3, "b3": db3, "gamma1": dgamma1, "beta1": dbeta1,
                "gamma2": dgamma2, "beta2": dbeta2}

    # ------------------------------------------------------------------ #
    def fit(self, X, y, lr=0.3, epochs=500, batch_size=0,
            window=None, stride=None, on_epoch=None, stop_flag=None):
        """全批量/小批量训练。

        X: (N, C, T)   y: (N,) 0/1
        on_epoch(epoch, loss, acc, lr) 每轮回调（用于 GUI 刷新）
        stop_flag: 可选 callable，返回 True 时中断训练
        返回 weights_dict
        """
        if window is not None:
            self.window = window
        if self.params is None:
            self.init_params()
        p = self.params
        N = X.shape[0]
        eye = np.eye(2)
        bs = batch_size if 0 < batch_size <= N else N

        for ep in range(epochs):
            if stop_flag is not None and stop_flag():
                break
            # mini-batch 索引
            perm = self._rng.permutation(N) if bs < N else np.arange(N)
            n_batches = int(np.ceil(N / bs))
            ep_loss = 0.0
            ep_acc = 0.0
            for bi in range(n_batches):
                idx = perm[bi * bs:(bi + 1) * bs]
                xb = X[idx]
                yb = y[idx]
                m = xb.shape[0]
                logits, c = self.forward(xb, training=True)
                probs = softmax(logits)
                loss = -np.mean(np.log(probs[np.arange(m), yb] + 1e-8))
                dlogits = (probs - eye[yb]) / m
                grads = self._backward(p, c, dlogits, m)
                for k, d in grads.items():
                    p[k] = (p[k] - lr * np.clip(d, -100, 100).astype(np.float32)).astype(np.float32)
                ep_loss += loss * m
                ep_acc += float((probs.argmax(1) == yb).mean()) * m
            ep_loss /= N
            ep_acc /= N

            if on_epoch is not None:
                on_epoch(ep, float(ep_loss), float(ep_acc), lr)

        return self.get_weights()

    # ------------------------------------------------------------------ #
    def get_weights(self):
        return {k: np.asarray(v) for k, v in self.params.items()}

    # ------------------------------------------------------------------ #
    def predict(self, X):
        logits, _ = self.forward(X, training=False)
        return softmax(logits).argmax(1), softmax(logits)

    # ------------------------------------------------------------------ #
    def export(self, out_dir, scaler=None):
        """导出与实时推理兼容的权重与标准化参数。

        weights: eeg_cnn_best_weights.npy (dict)
        scaler : scaler_mean.npy / scaler_scale.npy 形状 (1536,)=12*128
        """
        os.makedirs(out_dir, exist_ok=True)
        weights = self.get_weights()
        np.save(os.path.join(out_dir, "eeg_cnn_best_weights.npy"), weights)
        if scaler is not None:
            mean = np.asarray(scaler.mean)
            scale = np.asarray(scaler.scale)
            # 展开为 (C*T,) 以匹配 realtime_core 的 reshape(12,128)
            if mean.ndim == 2:
                mean = mean.reshape(-1)
                scale = scale.reshape(-1)
            np.save(os.path.join(out_dir, "scaler_mean.npy"), mean.astype(np.float32))
            np.save(os.path.join(out_dir, "scaler_scale.npy"), scale.astype(np.float32))
        return weights


def forward(weights, x, training=False):
    """与 EEG_Football_Realtime/train_model.py 完全等价的前向函数。

    推理端可以直接调用：logits, _ = forward(weights, x)
    """
    # 输入形状 (N, C, T)
    p = weights
    z1, _ = conv_forward(x, p["W1"], p["b1"])
    b1o, _ = bn_forward(z1, p["gamma1"], p["beta1"],
                        p["running_mean1"], p["running_var1"],
                        training=training)
    a1 = np.maximum(0, b1o)
    p1 = maxpool_forward(a1, 2)
    z2, _ = conv_forward(p1, p["W2"], p["b2"])
    b2o, _ = bn_forward(z2, p["gamma2"], p["beta2"],
                        p["running_mean2"], p["running_var2"],
                        training=training)
    a2 = np.maximum(0, b2o)
    p2 = maxpool_forward(a2, 2)
    g = p2.mean(-1)
    logits = g @ p["W3"] + p["b3"]
    return logits, {}