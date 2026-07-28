# -*- coding: utf-8 -*-
"""
config_store.py

全局配置持久化模块。把数据目录、模型导出路径等设置写入
eeggamev1.0/eeg_config.json，跨会话保持。

约束：
  - 仅依赖标准库 json / os / tempfile。
  - load_config：文件不存在、JSON 损坏、字段缺失 → 安全回退到默认值。
  - save_config：原子写（临时文件 + os.replace），避免中断损坏。
"""
import json
import os
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "eeg_config.json")

DEFAULT_CFG = {
    "move_dir": "",
    "stay_dir": "",
    "export_dir": "",
}


def load_config(path=CONFIG_PATH):
    """读取配置。文件缺失/损坏/字段缺失时返回默认值。"""
    if not os.path.exists(path):
        return dict(DEFAULT_CFG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_CFG)
        out = dict(DEFAULT_CFG)
        for k in DEFAULT_CFG:
            if k in data and isinstance(data[k], str):
                out[k] = data[k]
        return out
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return dict(DEFAULT_CFG)


def save_config(cfg, path=CONFIG_PATH):
    """原子写配置。先写临时文件再 os.replace，避免中断损坏。"""
    cfg = dict(cfg or {})
    out = {}
    for k in DEFAULT_CFG:
        out[k] = str(cfg.get(k, DEFAULT_CFG[k]) or "")
    out_dir = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".eeg_config_", suffix=".json",
                                dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return out
