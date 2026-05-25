#!/usr/bin/env python3
"""Publish OffboardControlMode + TrajectorySetpoint toward a given NED target.

Usage:
    python3 fly_to.py <north_m> <east_m> <down_m> [yaw_rad]

LOCAL NED is relative to PX4's EKF origin (set near spawn at boot). The script
also (re-)engages offboard mode and arms the vehicle defensively, so it works
whether the drone is fresh or mid-flight after the previous publisher died.
"""

import sys

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
)


def main(argv):
    if len(argv) < 4:
        print("Usage: fly_to.py <north_m> <east_m> <down_m> [yaw_rad]", file=sys.stderr)
        sys.exit(2)

    target_n = float(argv[1])
    target_e = float(argv[2])
    target_d = float(argv[3])
    target_yaw = float(argv[4]) if len(argv) > 4 else 0.0

    rclpy.init()
    node = rclpy.create_node("fly_to_setpoint")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    mode_pub = node.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
    sp_pub = node.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
    cmd_pub = node.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

    counter = [0]

    def now_us():
        return int(node.get_clock().now().nanoseconds / 1000)

    def publish_vehicle_command(command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = now_us()
        cmd_pub.publish(msg)

    def tick():
        ts = now_us()
        mode = OffboardControlMode()
        mode.position = True
        mode.timestamp = ts
        mode_pub.publish(mode)

        sp = TrajectorySetpoint()
        sp.position = [target_n, target_e, target_d]
        sp.yaw = target_yaw
        sp.timestamp = ts
        sp_pub.publish(sp)

        # PX4 needs ~1s of heartbeats before it accepts the offboard switch,
        # so defer the mode + arm commands until tick 10.
        if counter[0] == 10:
            publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            node.get_logger().info("Sent arm + offboard mode commands")
        counter[0] += 1

    node.create_timer(0.1, tick)
    node.get_logger().info(
        f"Flying to NED=(N={target_n}, E={target_e}, D={target_d}) yaw={target_yaw}"
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
