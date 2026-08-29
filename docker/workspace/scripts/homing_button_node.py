#!/usr/bin/env python3
"""Leader tact-button homing (state machine): long-press a leader joystick
button -> arms return to the tabletop start pose. Runs continuously and handles
repeated presses.

Listens on the leader's tact-trigger topic (std_msgs/String, values:
"left" / "right" / "left_long_time" / "right_long_time"). On the configured
event (default: long-press the LEFT joystick tact = "left_long_time", which
cyclo_intelligence leaves "reserved for future use") it runs a small,
non-blocking state machine driven by a timer:

    IDLE --(button)--> DISARMING --(disarm_wait)--> MOVING --(duration)-->
        REARMING --> IDLE            (REARMING skipped with --no-rearm)

  * DISARMING : publish /reactivate=false so nothing fights the move
                (vr_controller stops publishing arm commands when disarmed).
  * MOVING    : send ONE timed JointTrajectory per arm group; the follower's
                joint_trajectory_controllers interpolate over --duration.
  * REARMING  : publish /reactivate=true so re-engaging the leader resumes
                teleop with no keyboard. Control does NOT enable until the
                leader is aligned to the robot -- keep the alignment threshold
                (vr_controller startup_ref_pos_threshold / _ori_threshold_deg)
                TIGHT so it only re-engages when your leader is near the homed
                pose, instead of snapping the robot back to your leader's pose.

The button is only accepted in IDLE, so a press mid-homing is ignored (no
re-entrancy). Nothing blocks: the timer advances states, so the node keeps
listening for the next press forever.

By default only the ARMS are homed. Head and lift are streamed continuously by
the leader's joystick_controller at 100 Hz whenever the leader stack is up, so a
reset only sticks with the leader stack fully stopped -- enable then with
--include-head / --include-lift.

Run inside the cyclo_intelligence container (has /workspace/poses + Zenoh RMW):
    python3 /workspace/scripts/homing_button_node.py \
        --model-dir /workspace/poses/tabletop_manip

Target pose comes from initial_state.json in --model-dir (same file the reset
script uses), so there is a single source of truth for the tabletop pose.
"""
import argparse
import math
import json
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

# State machine states.
IDLE, DISARMING, MOVING, REARMING = "IDLE", "DISARMING", "MOVING", "REARMING"


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
        self.state = IDLE
        self.t_enter = 0.0          # monotonic time the current state was entered
        self._press_pending = False

        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.create_subscription(String, args.trigger_topic, self._trigger_cb, 10)
        self.reactivate_pub = self.create_publisher(Bool, args.reactivate_topic, 10)
        self.pubs = {
            g: self.create_publisher(JointTrajectory, topic, 10)
            for g, (topic, _) in self.groups.items()
        }
        self.create_timer(0.05, self._tick)  # 20 Hz state-machine clock

        self.get_logger().info(
            f"Homing button ready (state machine). Long-press '{args.trigger_event}' "
            f"on {args.trigger_topic} to home {list(self.groups)} over "
            f"{args.duration:.1f}s. Re-arm after: {args.rearm}.")

    # --- callbacks -------------------------------------------------------
    def _js_cb(self, msg):
        self.joint_state = dict(zip(msg.name, msg.position))

    def _trigger_cb(self, msg):
        if msg.data.strip() != self.args.trigger_event:
            return
        if self.state != IDLE:
            self.get_logger().warning(f"Busy ({self.state}); ignoring button.")
            return
        if self.joint_state is None:
            self.get_logger().error("No /joint_states yet; cannot home. Is the bringup up?")
            return
        self._press_pending = True  # honored by the timer in IDLE

    # --- state machine ---------------------------------------------------
    def _enter(self, state):
        self.state = state
        self.t_enter = time.monotonic()

    def _elapsed(self):
        return time.monotonic() - self.t_enter

    def _tick(self):
        if self.state == IDLE:
            if self._press_pending:
                self._press_pending = False
                self.reactivate_pub.publish(Bool(data=False))
                self.get_logger().info(
                    f"HOME pressed -> disarm ({self.args.reactivate_topic}=false).")
                self._enter(DISARMING)

        elif self.state == DISARMING:
            if self._elapsed() >= self.args.disarm_wait:
                self._send_home()
                self._enter(MOVING)

        elif self.state == MOVING:
            if self._elapsed() >= self.args.duration + self.args.settle:
                if self.args.rearm:
                    self._enter(REARMING)
                else:
                    self.get_logger().info(
                        "Homing complete; left DISARMED (--no-rearm). "
                        "Re-arm to teleop again.")
                    self._enter(IDLE)

        elif self.state == REARMING:
            self.reactivate_pub.publish(Bool(data=True))
            self.get_logger().info(
                f"Re-armed ({self.args.reactivate_topic}=true). Bring the leader to the "
                "home pose and re-engage; control enables only when it matches.")
            self._enter(IDLE)

    def _send_home(self):
        current = dict(self.joint_state)
        self.get_logger().info(f"Homing to tabletop pose over {self.args.duration:.1f}s:")
        for group, (topic, pred) in self.groups.items():
            names = [n for n in self.target if pred(n) and n in current]
            if not names:
                continue
            worst = max(abs(self.target[n] - current[n]) for n in names)
            self.get_logger().info(f"  {group}: {len(names)} joints, max |delta|={worst:.3f} rad")
            msg = JointTrajectory()
            msg.joint_names = names
            pt = JointTrajectoryPoint()
            pt.positions = [self.target[n] for n in names]
            pt.time_from_start.sec = int(math.floor(self.args.duration))
            pt.time_from_start.nanosec = int(
                (self.args.duration - math.floor(self.args.duration)) * 1e9)
            msg.points.append(pt)
            self.pubs[group].publish(msg)


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
    ap.add_argument("--settle", type=float, default=1.0,
                    help="extra wait after the move before re-arming (s)")
    ap.add_argument("--no-rearm", dest="rearm", action="store_false",
                    help="leave the controller disarmed after homing "
                         "(default: re-arm so re-engaging resumes without the keyboard; "
                         "keep the alignment threshold tight so it won't snap back)")
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
