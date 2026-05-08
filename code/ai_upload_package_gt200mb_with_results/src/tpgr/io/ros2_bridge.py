from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any


class Ros2JsonPublisher:
    def __init__(self, topic_name: str = "/perception/traffic_police_command") -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError as e:
            raise RuntimeError("ROS2 Python 环境不可用，请在已安装 rclpy 的环境中运行。") from e

        self._rclpy = rclpy
        self._String = String

        class _Node(Node):
            def __init__(self, topic_name: str, msg_type) -> None:
                super().__init__("tpgr_command_publisher")
                self.pub = self.create_publisher(msg_type, topic_name, 10)

        self.node = _Node(topic_name, self._String)

    def publish(self, message: dict[str, Any]) -> None:
        msg = self._String()
        msg.data = json.dumps(message, ensure_ascii=False)
        self.node.pub.publish(msg)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)
