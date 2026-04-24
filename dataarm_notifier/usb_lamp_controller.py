"""
Minimal USB lamp controller used by the pedal-controlled collection flow.
"""

import time
from enum import Enum


class LightColor(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    WHITE = "white"
    YELLOW = "yellow"
    CYAN = "cyan"
    MAGENTA = "magenta"
    OFF = "off"


class USBLampController:
    def __init__(self, port="/dev/ttyUSB1", baudrate=4800):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.register_map = {
            LightColor.RED: 0x0008,
            LightColor.GREEN: 0x0003,
            LightColor.BLUE: 0x0002,
            LightColor.WHITE: 0x0001,
        }
        self.default_pwm = 1999

    def _crc16(self, data):
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def _build_command(self, register_addr, value):
        command = bytes(
            [
                0x01,
                0x06,
                (register_addr >> 8) & 0xFF,
                register_addr & 0xFF,
                (value >> 8) & 0xFF,
                value & 0xFF,
            ]
        )
        return command + self._crc16(command)

    def _ensure_connection(self):
        if self.serial_conn is not None:
            return True

        try:
            import serial

            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity="N",
                stopbits=2,
                timeout=1,
            )
            return True
        except Exception as exc:
            print(f"[WARN] USB lamp unavailable on {self.port}: {exc}")
            self.serial_conn = None
            return False

    def _send_command(self, command):
        if not self._ensure_connection():
            return False

        try:
            self.serial_conn.write(command)
            self.serial_conn.flush()
            time.sleep(0.05)
            return True
        except Exception as exc:
            print(f"[WARN] Failed to write lamp command: {exc}")
            return False

    def set_light_on(self, on=True):
        value = 0x0001 if on else 0x0000
        return self._send_command(self._build_command(0x0004, value))

    def _set_color_brightness(self, color, brightness=100):
        pwm_value = int(self.default_pwm * brightness / 100)
        pwm_value = max(0, min(self.default_pwm, pwm_value))
        return self._send_command(self._build_command(self.register_map[color], pwm_value))

    def turn_off_all(self):
        self._set_color_brightness(LightColor.RED, 0)
        self._set_color_brightness(LightColor.GREEN, 0)
        self._set_color_brightness(LightColor.BLUE, 0)
        self._set_color_brightness(LightColor.WHITE, 0)

    def set_red(self, brightness=100):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.RED, brightness)

    def set_green(self, brightness=100):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.GREEN, brightness)

    def set_blue(self, brightness=100):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.BLUE, brightness)

    def set_white(self, brightness=100):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.WHITE, brightness)

    def set_yellow(self, brightness=100):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.RED, brightness)
        self._set_color_brightness(LightColor.GREEN, brightness)

    def set_cyan(self, brightness=50):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.GREEN, brightness)
        self._set_color_brightness(LightColor.BLUE, brightness)

    def set_magenta(self, brightness=100):
        self.turn_off_all()
        self.set_light_on(True)
        self._set_color_brightness(LightColor.RED, brightness)
        self._set_color_brightness(LightColor.BLUE, brightness)

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
