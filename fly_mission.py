#!/usr/bin/env python3
"""Execute a multi-waypoint mission read from a JSON file.

The mission file is a JSON document of the form::

    {
      "world": "empty_test",
      "waypoints": [
        {"label": "wp1", "n": 0,  "e": 0,  "d": -10, "yaw": 0.0, "hold_s": 2.0},
        {"label": "wp2", "n": 5,  "e": 5,  "d": -10, "yaw": 1.57, "hold_s": 3.0},
        ...
      ]
    }

Each waypoint is in PX4 LOCAL NED (m), yaw is radians, hold_s is how long to
linger after entering the tolerance ball before advancing.

Like fly_to.py, this also (re-)engages offboard mode and arms the vehicle
defensively so it works whether the drone is fresh or mid-flight.

Usage:
    python3 fly_mission.py <mission.json>
"""

import json
import os
import sys
import time

import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)


REACH_XY = 1.5   # m horizontal tolerance
REACH_Z = 1.0    # m vertical tolerance


def main(argv):
    if len(argv) < 2:
        print("Usage: fly_mission.py <mission.json>", file=sys.stderr)
        sys.exit(2)

    with open(argv[1]) as f:
        mission = json.load(f)
    waypoints = mission.get("waypoints", [])
    if not waypoints:
        print("mission has no waypoints", file=sys.stderr)
        sys.exit(1)

    # Swarm identity. DRONE_NS is the PX4 uXRCE-DDS namespace (e.g. "px4_1") so
    # this mission node drives that vehicle's topics; MAV_SYS_ID is the vehicle's
    # MAVLink system id (idx+1). Defaults reproduce the single-drone path:
    # empty namespace → legacy '/fmu/...' topics, sysid 1.
    drone_ns = os.getenv("DRONE_NS", "").strip("/")
    sys_id = int(os.getenv("MAV_SYS_ID", "1"))
    pfx = f"/{drone_ns}" if drone_ns else ""

    rclpy.init()
    node = rclpy.create_node(f"fly_mission_{sys_id}")
    pub_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    sub_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    mode_pub = node.create_publisher(OffboardControlMode, f"{pfx}/fmu/in/offboard_control_mode", pub_qos)
    sp_pub = node.create_publisher(TrajectorySetpoint, f"{pfx}/fmu/in/trajectory_setpoint", pub_qos)
    cmd_pub = node.create_publisher(VehicleCommand, f"{pfx}/fmu/in/vehicle_command", pub_qos)

    state = {"n": 0.0, "e": 0.0, "d": 0.0, "fix": False}

    def on_lpos(msg):
        state["n"] = msg.x
        state["e"] = msg.y
        state["d"] = msg.z
        state["fix"] = bool(msg.xy_valid and msg.z_valid)

    node.create_subscription(
        VehicleLocalPosition,
        f"{pfx}/fmu/out/vehicle_local_position",
        on_lpos,
        sub_qos,
    )

    counter = [0]
    wp_idx = [0]
    reached_at = [None]
    done = [False]

    def now_us():
        return int(node.get_clock().now().nanoseconds / 1000)

    def send_cmd(command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = sys_id
        msg.target_component = 1
        msg.source_system = sys_id
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = now_us()
        cmd_pub.publish(msg)

    def tick():
        ts = now_us()
        # Always hold on the last waypoint after completion (don't drop out of offboard).
        active_idx = min(wp_idx[0], len(waypoints) - 1)
        wp = waypoints[active_idx]

        mode = OffboardControlMode()
        mode.position = True
        mode.timestamp = ts
        mode_pub.publish(mode)

        sp = TrajectorySetpoint()
        sp.position = [float(wp["n"]), float(wp["e"]), float(wp["d"])]
        sp.yaw = float(wp.get("yaw", 0.0))
        sp.timestamp = ts
        sp_pub.publish(sp)

        if counter[0] == 10:
            send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            node.get_logger().info(
                f"Mission start — {len(waypoints)} waypoints. "
                f"first: {wp.get('label', '?')} "
                f"NED=({wp['n']:.2f}, {wp['e']:.2f}, {wp['d']:.2f}) "
                f"yaw={wp.get('yaw', 0):.2f}"
            )
        counter[0] += 1

        if done[0]:
            return
        if not state["fix"]:
            return

        dn = state["n"] - wp["n"]
        de = state["e"] - wp["e"]
        dd = state["d"] - wp["d"]
        in_zone = (
            abs(dn) < REACH_XY
            and abs(de) < REACH_XY
            and abs(dd) < REACH_Z
        )

        if in_zone:
            if reached_at[0] is None:
                reached_at[0] = time.time()
                node.get_logger().info(
                    f"Reached waypoint {wp_idx[0] + 1}/{len(waypoints)} "
                    f"({wp.get('label', '?')}) — holding {wp.get('hold_s', 0.0)}s"
                )
            elif time.time() - reached_at[0] >= float(wp.get("hold_s", 0.0)):
                wp_idx[0] += 1
                reached_at[0] = None
                if wp_idx[0] >= len(waypoints):
                    done[0] = True
                    node.get_logger().info(
                        "Mission complete — holding final setpoint"
                    )
        else:
            reached_at[0] = None

    node.create_timer(0.1, tick)
    node.get_logger().info(
        f"Loaded mission '{argv[1]}': {len(waypoints)} waypoints "
        f"(generator={mission.get('generator', '?')})"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
