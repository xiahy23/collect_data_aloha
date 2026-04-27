"""
State machine for lamp indication and Enter/pedal-based recording toggling.
"""

import glob
import os
import threading
from enum import Enum
from typing import Callable, Optional

from .keyboard_listener import KeyboardListener
from .usb_lamp_controller import USBLampController


class RobotState(Enum):
    IDLE = "idle"
    TEACH = "teach"
    SAVING = "saving"
    HOMING = "homing"
    EXECUTE_STOPPED = "execute_stopped"
    EXECUTE_RUNNING = "execute_running"
    ERROR = "error"


class RobotStateNotifier:
    def __init__(
        self,
        port: Optional[str] = None,
        auto_detect: bool = True,
        pedal_device: Optional[str] = None,
        trigger_key: str = "enter",
    ):
        self._state = RobotState.IDLE
        self._lamp = None
        self._keyboard = None
        self._enter_callback = None
        self._lock = threading.Lock()
        self._pedal_device = pedal_device
        self._trigger_key = trigger_key

        if port is None and auto_detect:
            if os.path.exists("/dev/dataarm_notifier"):
                port = "/dev/dataarm_notifier"
            else:
                port = self._auto_detect_port() or "/dev/ttyUSB1"

        if port:
            self._lamp = USBLampController(port=port)
            self._set_color_for_state(self._state)

    @staticmethod
    def _auto_detect_port():
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cu.usbserial*", "/dev/tty.usbserial*"]
        for pattern in patterns:
            ports = glob.glob(pattern)
            if ports:
                return sorted(ports)[0]
        return None

    def _set_color_for_state(self, state: RobotState):
        if self._lamp is None:
            print(f"[SIM] state={state.value}")
            return

        color_map = {
            RobotState.IDLE: self._lamp.set_cyan,
            RobotState.TEACH: self._lamp.set_green,
            RobotState.SAVING: self._lamp.set_yellow,
            RobotState.HOMING: self._lamp.set_magenta,
            RobotState.EXECUTE_STOPPED: self._lamp.set_blue,
            RobotState.EXECUTE_RUNNING: self._lamp.set_white,
            RobotState.ERROR: self._lamp.set_red,
        }
        color_map[state]()

    def set_state(self, state: RobotState):
        with self._lock:
            self._state = state
            self._set_color_for_state(state)

    def idle(self):
        self.set_state(RobotState.IDLE)

    def teach(self):
        self.set_state(RobotState.TEACH)

    def saving(self):
        self.set_state(RobotState.SAVING)

    def homing(self):
        self.set_state(RobotState.HOMING)

    def error(self):
        self.set_state(RobotState.ERROR)

    def on_enter_pressed(self, callback: Callable):
        self._enter_callback = callback

    def _handle_enter(self):
        if self._enter_callback:
            self._enter_callback()

    def start_keyboard_listener(self):
        if self._keyboard is None:
            self._keyboard = KeyboardListener(
                device_path=self._pedal_device,
                trigger_key=self._trigger_key,
            )
        self._keyboard.register_callback("enter", self._handle_enter)
        self._keyboard.start()

    def stop_keyboard_listener(self):
        if self._keyboard:
            self._keyboard.stop()

    def cleanup(self):
        self.stop_keyboard_listener()
        if self._lamp:
            self._lamp.turn_off_all()
            self._lamp.close()


class RecordingController:
    def __init__(
        self,
        port: Optional[str] = None,
        pedal_device: Optional[str] = None,
        trigger_key: str = "enter",
    ):
        self._notifier = RobotStateNotifier(
            port=port,
            pedal_device=pedal_device,
            trigger_key=trigger_key,
        )
        self._is_recording = False
        self._start_callback = None
        self._stop_callback = None
        self._lock = threading.Lock()

    @property
    def is_recording(self):
        return self._is_recording

    @property
    def notifier(self):
        return self._notifier

    def on_recording_start(self, callback: Callable):
        self._start_callback = callback

    def on_recording_stop(self, callback: Callable):
        self._stop_callback = callback

    def toggle_recording(self):
        with self._lock:
            if not self._is_recording:
                self._is_recording = True
                self._notifier.teach()
                if self._start_callback:
                    self._start_callback()
            else:
                self._is_recording = False
                self._notifier.saving()
                if self._stop_callback:
                    self._stop_callback()

    def start(self):
        self._notifier.on_enter_pressed(self.toggle_recording)
        self._notifier.start_keyboard_listener()
        self._notifier.idle()

    def stop(self):
        self._notifier.cleanup()
