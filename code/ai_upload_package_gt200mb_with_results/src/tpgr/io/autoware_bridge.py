from __future__ import annotations

import json
from typing import Any


class AutowareBridge:
    """演示性桥接器。

    推荐做法：
    1. 将本模块输出的 JSON 命令发布到 `/perception/traffic_police_command`
    2. 在 Autoware 内部转译为：
       - `external velocity limit`（STOP / SLOW_DOWN / KEEP_WAIT）
       - 或 planner 场景约束输入
    3. 再由 `vehicle_cmd_gate` / planner 统一仲裁
    """

    def __init__(self) -> None:
        self.velocity_limit_supported = False
        self.string_supported = False
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
            self._rclpy = rclpy
            self._String = String

            class _Node(Node):
                def __init__(self, msg_type) -> None:
                    super().__init__("tpgr_autoware_bridge")
                    self.json_pub = self.create_publisher(msg_type, "/perception/traffic_police_command", 10)

            self.node = _Node(self._String)
            self.string_supported = True

            try:
                from tier4_planning_msgs.msg import VelocityLimit
                self._VelocityLimit = VelocityLimit
                self.node.vel_pub = self.node.create_publisher(VelocityLimit, "/planning/scenario_planning/max_velocity_default", 10)
                self.velocity_limit_supported = True
            except Exception:
                self.velocity_limit_supported = False
        except ImportError as e:
            raise RuntimeError("需要 ROS2 + Autoware 环境。") from e

    def publish_json(self, message: dict[str, Any]) -> None:
        msg = self._String()
        msg.data = json.dumps(message, ensure_ascii=False)
        self.node.json_pub.publish(msg)

    def publish_velocity_hint(self, command: str) -> None:
        if not self.velocity_limit_supported:
            return
        msg = self._VelocityLimit()
        if command in {"STOP", "KEEP_WAIT", "LEFT_TURN_WAIT"}:
            msg.max_velocity = 0.0
        elif command == "SLOW_DOWN":
            msg.max_velocity = 3.0
        else:
            msg.max_velocity = 8.0
        self.node.vel_pub.publish(msg)
