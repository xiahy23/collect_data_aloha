"""
Keyboard-style pedal listener implemented with evdev.

This works in headless Linux environments and does not depend on X11/Wayland.
It can auto-discover keyboard-like input devices under /dev/input/by-id.
"""

import glob
import os
import threading
from typing import Callable, Optional

try:
    from evdev import InputDevice, ecodes

    EVDEV_AVAILABLE = True
except ImportError as exc:
    EVDEV_AVAILABLE = False
    _evdev_error = exc


DEFAULT_PEDAL_DEVICE = "/dev/input/by-id/usb-0483_5750-if01-event-kbd"
DEFAULT_SCAN_GLOB = "/dev/input/by-id/*event-kbd"
KEY_NAME_TO_CODE = {
    "enter": "KEY_ENTER",
    "space": "KEY_SPACE",
    "esc": "KEY_ESC",
    "escape": "KEY_ESC",
    "f13": "KEY_F13",
    "f24": "KEY_F24",
}


class KeyboardListener:
    def __init__(self, device_path: Optional[str] = None, trigger_key: str = "enter"):
        self.device_path = device_path or os.environ.get("DATAARM_PEDAL_DEVICE")
        self.trigger_key = trigger_key.lower()
        self.trigger_code = self._resolve_trigger_code(self.trigger_key)
        self._listening = False
        self._thread: Optional[threading.Thread] = None
        self._device: Optional[InputDevice] = None
        self._callbacks = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_trigger_code(trigger_key: str):
        attr_name = KEY_NAME_TO_CODE.get(trigger_key.lower(), f"KEY_{trigger_key.upper()}")
        if not hasattr(ecodes, attr_name):
            raise ValueError(f"Unsupported trigger key: {trigger_key}")
        return getattr(ecodes, attr_name)

    @staticmethod
    def _candidate_device_paths():
        candidates = []
        by_id = sorted(glob.glob(DEFAULT_SCAN_GLOB))
        candidates.extend(by_id)
        if DEFAULT_PEDAL_DEVICE not in candidates:
            candidates.append(DEFAULT_PEDAL_DEVICE)
        return candidates

    def _pick_device_path(self):
        if self.device_path:
            return self.device_path

        candidates = self._candidate_device_paths()
        if not candidates:
            raise RuntimeError("No keyboard-like input devices found under /dev/input/by-id")

        for path in candidates:
            try:
                device = InputDevice(path)
                name = getattr(device, "name", "")
                print(f"[INFO] pedal candidate: {path} ({name})")
                device.close()
                return path
            except PermissionError as exc:
                raise PermissionError(
                    f"Permission denied for pedal device {path}. "
                    "Grant read access to the input device or configure udev rules."
                ) from exc
            except Exception:
                continue

        raise RuntimeError(
            "No readable pedal device found. "
            "Set DATAARM_PEDAL_DEVICE or pass --pedal_device explicitly."
        )

    def register_callback(self, key: str, callback: Callable):
        with self._lock:
            self._callbacks[key.lower()] = callback

    def unregister_callback(self, key: str):
        with self._lock:
            self._callbacks.pop(key.lower(), None)

    def _dispatch_key(self, key_name: str):
        with self._lock:
            callback = self._callbacks.get(key_name.lower())
        if callback:
            callback()

    def _read_loop(self):
        try:
            resolved_path = self._pick_device_path()
        except Exception as exc:
            print(f"[ERROR] pedal: cannot find device: {exc}")
            print("[HINT]  Run: sudo usermod -aG input $USER  then log out/in")
            self._listening = False
            return
        try:
            self._device = InputDevice(resolved_path)
            self.device_path = resolved_path
            # Grab the device exclusively so pedal key events (KEY_ENTER) do
            # not also propagate to whatever window has UI focus.  Without
            # this, pressing the pedal while a Tk button has focus can
            # accidentally activate that button or insert newlines into the
            # focused terminal, which makes the pedal feel "broken" after
            # any UI interaction.
            try:
                self._device.grab()
                print(f"[INFO] pedal device grabbed exclusively: {resolved_path}")
            except Exception as grab_exc:
                print(f"[WARN] could not grab pedal device exclusively: {grab_exc}")
            print(
                f"[INFO] pedal listener attached to {resolved_path} "
                f"(trigger key: {self.trigger_key})"
            )
            for event in self._device.read_loop():
                if not self._listening:
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                if event.value != 1:
                    continue
                if event.code == self.trigger_code:
                    self._dispatch_key(self.trigger_key)
        except Exception as exc:
            print(f"[WARN] pedal listener stopped: {exc}")
        finally:
            if self._device is not None:
                try:
                    self._device.ungrab()
                except Exception:
                    pass
                try:
                    self._device.close()
                except Exception:
                    pass
                self._device = None

    def start(self):
        if not EVDEV_AVAILABLE:
            raise RuntimeError(f"evdev not available: {_evdev_error}. Install with: pip install evdev")
        if self._listening:
            return
        self._listening = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._listening = False
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_listening(self) -> bool:
        return self._listening
