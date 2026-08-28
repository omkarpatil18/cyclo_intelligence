#!/usr/bin/env python3
"""Leader tact-button homing: press a leader joystick button -> arms return to
the tabletop start pose.

Listens on the leader's tact-trigger topic (std_msgs/String, values:
"left" / "right" / "left_long_time" / "right_long_time"). On the configured
event (default: long-press the LEFT joystick tact = "left_long_time", which
cyclo_intelligence leaves "reserved for future use") it:

  1. publishes /reactivate = false  (belt-and-suspenders: stops the cyclo
     leader-mode controller streaming to the arm topics; a no-op if already
     disarmed / not teleoperating),
  2. sends ONE timed JointTrajectory per arm group so the follower's
     joint_trajectory_controllers interpolate smoothly to the target over
     --duration, exactly like go_to_initial_state.py.

By default only the ARMS are homed. Head and lift are driven continuously by
the leader's joystick_controller at 100 Hz whenever the leader stack is up, so
homing them only sticks if the leader is fully stopped -- enable with
--include-head / --include-lift only then.

Run inside the cyclo_intelligence container (has /workspace/poses + Zenoh RMW):
    python3 /workspace/scripts/homing_button_node.py \
        --model-dir /workspace/poses/tabletop_manip

Target pose comes from initial_state.json in --model-dir (same file the reset
script uses), so there is a single source of truth for the tabletop pose.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# group: (command topic, joint-name predicate). Same topics as go_to_initial_state.py.
GROUPS = {
    "arm_left": (
        "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
        lambda n: n.startswith(("arm_l_", "gripper_l_")),
    ),
    "arm_right": (
        "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
        lambda n: n.startswith(("arm_r_", "gripper_r_")),
    ),
    "head": (
        "/leader/joystick_controller_left/joint_trajectory",
        lambda n: n.startswith("head_"),
    ),
    "lift": (
        "/leader/joystick_controller_right/joint_trajectory",
        lambda n: n == "lift_joint",
    ),
}


def load_target(model_dir: Path) -> dict:
    with open(model_dir / "initial_state.json") as f:
        d = json.load(f)
    names, mean = d["joint_names"], d["initial_state_mean"]
    if len(names) != len(mean):
        raise ValueError("joint_names / initial_state_mean length mismatch")
    return dict(zip(names, mean))


class HomingButton(Node):
    def __init__(self, args):
        super().__init__("homing_button")
        self.args = args
        self.target = load_target(args.model_dir)

        self.groups = {"arm_left": GROUPS["arm_left"], "arm_right": GROUPS["arm_right"]}
        if args.include_head:
            self.groups["head"] = GROUPS["head"]
        if args.include_lift:
            self.groups["lift"] = GROUPS["lift"]

        self.joint_state = None
        self._busy = False

        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.create_subscription(String, args.trigger_topic, self._trigger_cb, 10)

        self.reactivate_pub = self.create_publisher(Bool, args.reactivate_topic, 10)
        self.pubs = {
            g: self.create_publisher(JointTrajectory, topic, 10)
            for g, (topic, _) in self.groups.items()
        }

        homed = ", ".join(self.groups)
        self.get_logger().info(
            f"Homing button ready. Press '{args.trigger_event}' on "
            f"{args.trigger_topic} to home [{homed}] over {args.duration:.1f}s.")

    def _js_cb(self, msg):
        self.joint_state = dict(zip(msg.name, msg.position))

    def _trigger_cb(self, msg):
        event = msg.data.strip()
        if event != self.args.trigger_event:
            return
        if self._busy:
            self.get_logger().warning("Homing already in progress; ignoring trigger.")
            return
        if self.joint_state is None:
            self.get_logger().error("No /joint_states yet; cannot home. Is the bringup up?")
            return
        self._home()

    def _home(self):
        self._busy = True
        try:
            # 1) Disarm the cyclo controller (safety belt; no-op if not teleoperating).
            self.reactivate_pub.publish(Bool(data=False))
            self.get_logger().info(f"Published {self.args.reactivate_topic}=false (disarm).")
            self._spin_for(self.args.disarm_wait)

            current = dict(self.joint_state)

            # 2) Build + publish one timed trajectory per group.
            self.get_logger().info(
                f"Homing to tabletop pose over {self.args.duration:.1f}s:")
            for group, (topic, pred) in self.groups.items():
                names = [n for n in self.target if pred(n) and n in current]
                if not names:
                    continue
                deltas = [self.target[n] - current[n] for n in names]
                worst = max(abs(d) for d in deltas)
                self.get_logger().info(
                    f"  {group}: {len(names)} joints, max |delta|={worst:.3f} rad")
                msg = JointTrajectory()
                msg.joint_names = names
                pt = JointTrajectoryPoint()
                pt.positions = [self.target[n] for n in names]
                pt.time_from_start.sec = int(math.floor(self.args.duration))
                pt.time_from_start.nanosec = int(
                    (self.args.duration - math.floor(self.args.duration)) * 1e9)
                msg.points.append(pt)
                self.pubs[group].publish(msg)

            self._spin_for(self.args.duration + 1.0)

            # 3) Re-arm automatically so re-engaging the leader resumes teleop
            #    with no keyboard. This only puts the controller back into its
            #    "waiting for reference alignment" state; control stays disabled
            #    (arm held at the tabletop pose) until the leader is re-engaged
            #    and its pose matches the robot.
            if self.args.rearm:
                self.reactivate_pub.publish(Bool(data=True))
                self.get_logger().info(
                    f"Published {self.args.reactivate_topic}=true (re-armed). "
                    "Re-engage the leader to resume teleop.")
            else:
                self.get_logger().info("Homing complete (left disarmed; --no-rearm).")
        finally:
            self._busy = False

    def _spin_for(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/poses/tabletop_manip"),
                    help="dir containing initial_state.json (default: tabletop_manip)")
    ap.add_argument("--trigger-topic", default="/leader/joystick_controller/tact_trigger")
    ap.add_argument("--trigger-event", default="left_long_time",
                    choices=["left", "right", "left_long_time", "right_long_time"],
                    help="tact event that fires homing (default: left_long_time)")
    ap.add_argument("--reactivate-topic", default="/reactivate")
    ap.add_argument("--duration", type=float, default=5.0, help="seconds for the move")
    ap.add_argument("--disarm-wait", type=float, default=0.4,
                    help="pause after disarm before commanding (s)")
    ap.add_argument("--no-rearm", dest="rearm", action="store_false",
                    help="leave the controller disarmed after homing "
                         "(default: re-arm so re-engaging the leader resumes teleop)")
    ap.add_argument("--include-head", action="store_true",
                    help="also home the head (only sticks if the leader stack is stopped)")
    ap.add_argument("--include-lift", action="store_true",
                    help="also home the lift (only sticks if the leader stack is stopped)")
    args = ap.parse_args()

    rclpy.init()
    node = HomingButton(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
