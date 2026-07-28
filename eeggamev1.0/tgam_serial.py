# -*- coding: utf-8 -*-
"""
TGAM 脑电传感器串口协议解析（Python / pyserial）

移植自 student-boilerplate/src/hooks/useTGAMSerial.ts 的 Web Serial 解析逻辑，
用于在桌面端用 pyserial 实时读取 TGAM（NeuroSky ThinkGear ASIC Module）输出的
原始脑电波形 (RAW EEG)。

协议要点：
  - 波特率 57600，8N1
  - 同步头: 0xAA 0xAA
  - 小包（RAW 波形）: AA AA 04 80 02 [high] [low] [checksum]
        校验和: ((0x80 + 0x02 + high + low) ^ 0xFFFF) & 0xFF
        RAW = (high<<8)|low; 若 > 32768 则减去 65536（有符号 16 位）
  - 大包（频段/专注/放松）: AA AA 20 02 ... 见 parse_large()

使用方法：
    reader = TGAMReader(port="COM3", on_raw=callback)
    reader.open()
    ... callback(raw_value) 被逐样本调用 ...
    reader.close()
"""

import threading
import time

try:
    import serial
    import serial.tools.list_ports
except Exception as exc:  # pyserial 未装时给出友好提示
    serial = None
    serial_tools = None
    _IMPORT_ERR = exc


def list_ports():
    """返回可用串口列表（如 ['COM3', 'COM4']）。"""
    if serial is None:
        return []
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def _parse_hex_byte(hex_str, start, length=2):
    return int(hex_str[start:start + length], 16)


class TGAMReader:
    """TGAM 串口读取与协议解析（后台线程）。"""

    def __init__(self, port, baud=57600, on_raw=None, on_packet=None, on_status=None):
        """
        :param port: 串口号，如 'COM3'
        :param baud: 波特率，默认 57600（TGAM）
        :param on_raw: 回调(raw_value:int)，每解析到一个 RAW 样本调用一次
        :param on_packet: 回调(data:dict)，每解析到一个大包(频段/专注/放松)调用
        :param on_status: 回调(msg:str, level:str)，状态/日志
        """
        if serial is None:
            raise ImportError(
                "未安装 pyserial，请先执行: pip install pyserial\n原始错误: %r" % _IMPORT_ERR
            )
        self.port_name = port
        self.baud = baud
        self.on_raw = on_raw
        self.on_packet = on_packet
        self.on_status = on_status
        self.ser = None
        self._running = False
        self._thread = None
        self._hex_buf = ""  # 十六进制字符串缓存，模拟 TS 端的 rawHexStr

    # ------------------------------------------------------------------ #
    def open(self):
        """打开串口并启动后台读取线程。"""
        if self._running:
            return
        self.ser = serial.Serial(self.port_name, self.baud, timeout=0.2)
        # 与浏览器端一致：拉高 DTR / RTS（部分 USB-TTL 模块需要）
        try:
            self.ser.setDTR(True)
            self.ser.setRTS(True)
        except Exception:
            pass
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        if self.on_status:
            self.on_status("已打开串口 %s @ %d" % (self.port_name, self.baud), "ok")

    def is_open(self):
        return self._running and self.ser is not None and self.ser.is_open

    def close(self):
        self._running = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        if self.on_status:
            self.on_status("串口已关闭", "info")

    # ------------------------------------------------------------------ #
    def _read_loop(self):
        """后台线程：读字节 -> 累积 hex 缓存 -> 定位 AAAA 同步 -> 解析小包/大包。"""
        while self._running:
            try:
                data = self.ser.read(64)
            except Exception as e:
                if self._running and self.on_status:
                    self.on_status("读串口出错: %s" % e, "err")
                time.sleep(0.05)
                continue
            if not data:
                continue

            # 把新字节追加到十六进制缓存
            self._hex_buf += "".join("%02x" % b for b in data)

            # 限制缓存长度，避免无限增长
            if len(self._hex_buf) > 4096:
                self._hex_buf = self._hex_buf[-2048:]

            self._parse_buffer()

    def _parse_buffer(self):
        """解析 hex 缓存中的小包(AAAA048002)与大包(AAAA2002)。"""
        while True:
            idx_small = self._hex_buf.find("aaaa048002")
            idx_large = self._hex_buf.find("aaaa2002")

            # 没有更多可解析的帧
            if idx_small == -1 and idx_large == -1:
                break

            # 选择最靠前的帧
            if idx_small != -1 and (idx_large == -1 or idx_small < idx_large):
                start = idx_small
                frame = "small"
            else:
                start = idx_large
                frame = "large"

            if frame == "small":
                # 小包总长 16 个 hex 字符 (8 字节)
                if len(self._hex_buf) < start + 16:
                    break
                chunk = self._hex_buf[start:start + 16]
                high = _parse_hex_byte(chunk, 10, 2)
                low = _parse_hex_byte(chunk, 12, 2)
                checksum = _parse_hex_byte(chunk, 14, 2)
                calc = ((0x80 + 0x02 + high + low) ^ 0xFFFF) & 0xFF
                # 移除已处理部分
                self._hex_buf = self._hex_buf[start + 16:]
                if calc != checksum:
                    if self.on_status:
                        self.on_status("RAW 校验失败，丢弃", "warn")
                    continue
                raw = (high << 8) | low
                if raw > 32768:
                    raw -= 65536
                if self.on_raw:
                    self.on_raw(raw)

            else:  # large
                # 大包: AA AA 20 02 [length] [payload...]
                if len(self._hex_buf) < start + 10:
                    break
                payload_len = _parse_hex_byte(self._hex_buf, start + 8, 2)
                total = 10 + payload_len * 2  # 头 10 hex + payload
                if len(self._hex_buf) < start + total:
                    break
                payload_hex = self._hex_buf[start + 10:start + total]
                self._hex_buf = self._hex_buf[start + total:]
                data = self._parse_large(payload_hex)
                if data and self.on_packet:
                    self.on_packet(data)

    @staticmethod
    def _parse_large(payload_hex):
        """解析大包载荷（标准 TGAM 协议）。

        返回 dict: {signalQuality, attention, meditation,
                    eegPower:{delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,lowGamma,midGamma}}
        频段为 3 字节大端无符号整数（标准 TGAM）。
        """
        data = {"eegPower": {}, "attention": None, "meditation": None,
                "signalQuality": None}
        i = 0
        n = len(payload_hex)
        while i + 2 <= n:
            code = _parse_hex_byte(payload_hex, i, 2)
            i += 2
            if code == 0x02:  # 信号质量
                if i + 2 <= n:
                    data["signalQuality"] = _parse_hex_byte(payload_hex, i, 2)
                    i += 2
            elif code == 0x04:  # 专注度
                if i + 2 <= n:
                    data["attention"] = _parse_hex_byte(payload_hex, i, 2)
                    i += 2
            elif code == 0x05:  # 放松度
                if i + 2 <= n:
                    data["meditation"] = _parse_hex_byte(payload_hex, i, 2)
                    i += 2
            elif code == 0x83:  # 八频段脑电功率（每频段 3 字节大端）
                if i + 48 <= n:
                    names = ["delta", "theta", "lowAlpha", "highAlpha",
                             "lowBeta", "highBeta", "lowGamma", "midGamma"]
                    for name in names:
                        v = (_parse_hex_byte(payload_hex, i, 2) << 16) | \
                            (_parse_hex_byte(payload_hex, i + 2, 2) << 8) | \
                            _parse_hex_byte(payload_hex, i + 4, 2)
                        data["eegPower"][name] = v
                        i += 6
            else:
                # 未知 code：跳过其长度字段 + 数据
                if i + 2 <= n:
                    length = _parse_hex_byte(payload_hex, i, 2)
                    i += 2 + length * 2
        return data

    @staticmethod
    def build_feature_vector(packet):
        """从大包解析结果构造 12 通道特征向量（与训练 CSV 列顺序一致）：
        [signalQuality, attention, meditation, delta, theta, lowAlpha,
         highAlpha, lowBeta, highBeta, lowGamma, midGamma, 0]
        """
        ep = packet.get("eegPower", {})
        vec = [
            float(packet.get("signalQuality") or 0),
            float(packet.get("attention") or 0),
            float(packet.get("meditation") or 0),
            float(ep.get("delta", 0)),
            float(ep.get("theta", 0)),
            float(ep.get("lowAlpha", 0)),
            float(ep.get("highAlpha", 0)),
            float(ep.get("lowBeta", 0)),
            float(ep.get("highBeta", 0)),
            float(ep.get("lowGamma", 0)),
            float(ep.get("midGamma", 0)),
            0.0,  # 第 12 通道（训练时为全零 label 列）
        ]
        return vec


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # 简单自检：列出端口
    print("可用串口:", list_ports())
    print("TGAMReader 模块就绪。")