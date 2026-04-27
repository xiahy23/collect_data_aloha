# -- coding: UTF-8
"""
Web preview for three ROS camera topics.

Default topics:
    /camera_f/color/image_raw
    /camera_l/color/image_raw
    /camera_r/color/image_raw

Usage:
    python preview_three_cameras_web.py

Open in browser:
    http://<robot-ip>:8000/
"""

import argparse
import threading
import time
from http import server
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Three Camera Preview</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #101418;
      color: #e6edf3;
    }
    .wrap {
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 24px;
    }
    .meta {
      margin-bottom: 16px;
      color: #8b949e;
      font-size: 14px;
    }
    .panel {
      border: 1px solid #30363d;
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    img {
      display: block;
      width: 100%;
      height: auto;
    }
    .tips {
      margin-top: 12px;
      color: #8b949e;
      font-size: 13px;
    }
    code {
      background: #161b22;
      padding: 2px 6px;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>三路相机实时预览</h1>
    <div class="meta">页面会自动刷新 MJPEG 视频流。缺失相机会显示 waiting。</div>
    <div class="panel">
      <img src="/stream.mjpg" alt="three camera preview">
    </div>
    <div class="tips">
      浏览器地址: <code>/</code>，视频流地址: <code>/stream.mjpg</code>
    </div>
  </div>
</body>
</html>
"""


class CameraPreviewWeb:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frames = {
            "front": None,
            "left": None,
            "right": None,
        }
        self.stamps = {
            "front": 0.0,
            "left": 0.0,
            "right": 0.0,
        }

        rospy.init_node("preview_three_cameras_web", anonymous=True)
        rospy.Subscriber(args.img_front_topic, Image, self._make_cb("front"), queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(args.img_left_topic, Image, self._make_cb("left"), queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(args.img_right_topic, Image, self._make_cb("right"), queue_size=1, tcp_nodelay=True)

    def _make_cb(self, name):
        def _callback(msg):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError:
                return
            with self.lock:
                self.frames[name] = frame
                self.stamps[name] = time.time()

        return _callback

    def _placeholder(self, title):
        canvas = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            f"{title}: waiting",
            (20, self.args.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _prepare_panel(self, name, title):
        with self.lock:
            frame = self.frames[name]
            stamp = self.stamps[name]

        if frame is None:
            panel = self._placeholder(title)
        else:
            panel = cv2.resize(frame, (self.args.width, self.args.height))
            age = time.time() - stamp
            cv2.putText(
                panel,
                f"{title} age={age:.2f}s",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return panel

    def get_composite_jpeg(self):
        front = self._prepare_panel("front", "front")
        left = self._prepare_panel("left", "left")
        right = self._prepare_panel("right", "right")
        canvas = np.hstack([front, left, right])
        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("failed to encode preview image")
        return encoded.tobytes()


class ThreadedHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True


def make_handler(preview, fps):
    class PreviewHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                interval = 1.0 / float(max(fps, 1))
                while True:
                    if rospy.is_shutdown():
                        break
                    try:
                        frame = preview.get_composite_jpeg()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(interval)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                return

            self.send_error(404)

        def log_message(self, fmt, *args):
            return

    return PreviewHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Web preview for three ROS camera streams.")
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--width", type=int, default=480, help="Panel width.")
    parser.add_argument("--height", type=int, default=360, help="Panel height.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--display_fps", type=int, default=10, help="MJPEG refresh rate.")
    parser.add_argument("--jpeg_quality", type=int, default=80, help="Preview JPEG quality.")
    return parser.parse_args()


def main():
    args = parse_args()
    preview = CameraPreviewWeb(args)
    handler = make_handler(preview, args.display_fps)
    httpd = ThreadedHTTPServer((args.host, args.port), handler)
    print(f"Preview web server started: http://{args.host}:{args.port}/")
    print("Press Ctrl+C to quit.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
