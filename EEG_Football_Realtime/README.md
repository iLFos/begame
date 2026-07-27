# 实时脑电检测控制台（TGAM 串口协议）

基于 TGAM 脑电传感器串口协议，实时读取原始脑电 (RAW)，缓冲到 128 样本即做
滑窗推理，判定「运动 / 静止」，把结果 (1/0) 实时写入 `movda.txt`，由足球游戏
服务器 (`server.js`) 驱动红方球员前进。

> 本文件夹为**新建**实现，未改动原 `EEG_Football`、`student-boilerplate` 任何文件。

## 目录结构
```
EEG_Football_Realtime/
├── realtime_eeg_gui.py   # 主程序(Tkinter 实时控制台)
├── tgam_serial.py        # TGAM 串口协议解析(pyserial, 57600, AAAA 小包校验)
├── realtime_core.py      # 实时滑窗推理核心 + 离线 CSV 回放
├── train_model.py        # 已验证的 EEGCNN 前向(复用)
├── eeg_cnn_best_weights.npy / scaler_mean.npy / scaler_scale.npy
├── server.js / demov1.0.html / start.bat   # 足球游戏 + 信号服务器
├── movda.txt             # 控制信号文件(推理实时写 0/1)
├── requirements.txt / README.md
└── sample_data/          # 示例脑电 CSV(离线回放演示)
```

## 传感器协议（TGAM / NeuroSky ThinkGear）
- 串口：57600 波特，8N1，DTR/RTS 拉高
- 同步头 `AA AA`
- 小包(RAW 波形)：`AA AA 04 80 02 [high] [low] [checksum]`
  - 校验和：`((0x80+0x02+high+low) ^ 0xFFFF) & 0xFF`
  - RAW = 有符号 16 位（>32768 减 65536），约 512Hz
- 大包(频段/专注/放松)：`AA AA 20 02 ...`

## 使用方式
```
pip install -r requirements.txt
python realtime_eeg_gui.py
```
界面操作：
1. 选「串口」(COM) 与「波特率」(默认 57600)，点 **▶ 连接传感器** 开始实时检测；
2. 实时波形 + 运动/静止概率 + 红/绿信号灯随信号刷新；结果实时写入 `movda.txt`；
3. 点 **🎮 启动游戏** 打开 `http://localhost:8080`，红方随「运动」信号前进；
4. 无传感器时，点 **▶ 离线CSV回放** 选 `sample_data/*.csv` 验证闭环（已训模型对
   move 文件判运动、stay 文件判静止）。

## 关于通道
TGAM 实时输出**单通道** RAW，现有 CNN 输入 **12 通道**。本程序将单通道广播到
12 通道以复用已训权重（演示可用；真实 TGAM 数据分类精度不保证）。若你有单通道
带标注的 move/stay 数据，可改训 1 通道模型并替换 `realtime_core.py` 中广播逻辑。

## 依赖说明
- Python 3.14 + numpy/pandas/matplotlib（已装），新增 `pyserial`（串口读取）
- Node.js（运行 `server.js` 游戏服务器）
- 界面与文件读写统一 UTF-8 + SimHei 中文字体
