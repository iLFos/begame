# eeggamev1.0

EEG 脑电信号「移动 / 静止」AI 训练与推理控制一体化程序（v1.0）。

白色简约风格单窗口 Tkinter 应用；窗口顶部下拉切换两个功能模块：

1. **AI 训练程序**：自定义 epochs / 卷积核数 f1（f2 自动 = 2×f1），导入 move / stay CSV
   样本，自定义模型导出路径，实时查看 loss / acc 曲线。
2. **AI 推理检测程序**：加载训练好的模型，支持 **TGAM 串口** 与 **离线 CSV 回放**
   两种数据源；通过 **1-100 阈值滑块** 控制运动概率判定；可一键启动 node 游戏
   `server.js`，当模型输出 `prob[1] ≥ 阈值/100` 时，程序向 `movda.txt` 写入 `1`，
   驱动足球游戏中的「红方」向前移动。

## 快速开始

```cmd
:: 安装依赖
pip install -r requirements.txt

:: 启动主程序
python main.py

:: 单独启动游戏服务器（在主程序内点「启动游戏」亦可）
start.bat
```

启动主程序后，在「AI 推理检测程序」中加载模型，点击「启动游戏」即可在浏览器中
打开 `http://localhost:8080`（脚本会自动打开）。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `main.py` | 主入口，单窗口，下拉切换 |
| `train_panel.py` | 训练面板 |
| `infer_panel.py` | 推理面板 |
| `model_ops.py` | 纯 NumPy 算子 |
| `model.py` | `EEGCNN(f1, f2, k1, k2)` 可配置网络 |
| `data_utils.py` | 数据加载 / 概览 |
| `realtime_core.py` | 实时推理核心（含阈值判定） |
| `tgam_serial.py` | TGAM 串口协议解析 |
| `server.js` | 游戏服务器 |
| `demov1.0.html` | 足球游戏页面 |
| `movda.txt` | 控制信号文件（程序实时写入 0/1） |
| `run.bat` | 一键启动主程序 |
| `start.bat` | 一键启动游戏服务器 |

## 数据约定

- 输入形状：`(N, C=12, T=128)`
- CSV 列：`timestamp, signalQuality, attention, meditation, delta, theta, lowAlpha,
  highAlpha, lowBeta, highBeta, lowGamma, midGamma, label`
- 标签约定：`move/` 目录 → 1，`stay/` 目录 → 0

## 阈值逻辑

```python
thr = slider_value / 100.0           # slider ∈ [1,100]
is_motion = (prob[1] >= thr)         # prob[1] 为 softmax 输出的运动概率
write_signal(1 if is_motion else 0)
```

## 许可

内部使用。