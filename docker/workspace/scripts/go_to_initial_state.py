#!/usr/bin/env python3
"""Move the AI Worker (FFW SG2) to a policy's recorded initial state.

Reads ``initial_state.json`` next to a trained checkpoint and sends ONE timed
JointTrajectory per group (left arm, right arm, head; lift optional) so the
follower's joint_trajectory_controllers interpolate smoothly over ``--duration``.

Run inside the cyclo_intelligence container (rclpy + Zenoh RMW):
    python3 /workspace/scripts/go_to_initial_state.py \
        --model-dir /workspace/model/lerobot/ffw_sg2_wave-right_diffusion_state

Safety:
  * refuses to run if any OTHER node already publishes on a target topic
    (movej, leader broadcaster, or a running policy) — one command source only;
  * prints the per-joint move it is about to make and asks for confirmation
    (``--yes`` to skip);
  * after the move, reports per-joint error vs. the demo mean and exits
    non-zero if any joint is farther than ``--tolerance``.
It never publishes /cmd_vel.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

GROUPS = {
    # group: (topic, joint-name predicate)
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


def load_initial_state(model_dir: Path) -> dict:
    with open(model_dir / "initial_state.json") as f:
        d = json.load(f)
    names = d["joint_names"]
    mean = d["initial_state_mean"]
    if len(names) != len(mean):
        raise ValueError("joint_names / initial_state_mean length mismatch")
    return dict(zip(names, mean))


class GoToInitialState(Node):
    def __init__(self):
        super().__init__("go_to_initial_state")
        self.joint_state = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

    def _js_cb(self, msg):
        self.joint_state = dict(zip(msg.name, msg.position))

    def wait_joint_states(self, timeout=5.0):
        t0 = time.time()
        while self.joint_state is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.joint_state is not None

    def spin_for(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path, help="checkpoint dir containing initial_state.json")
    ap.add_argument("--duration", type=float, default=5.0, help="seconds for the timed move (default 5)")
    ap.add_argument("--tolerance", type=float, default=0.24, help="per-joint pass threshold in rad (default 0.24)")
    ap.add_argument("--include-lift", action="store_true", help="also command the lift (default: skip)")
    ap.add_argument("--skip-head", action="store_true", help="do not command the head")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, publish nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--strict", action="store_true",
                    help="abort on ANY other publisher, even idle ones (default: idle publishers only warn)")
    args = ap.parse_args()

    target = load_initial_state(args.model_dir)
    groups = dict(GROUPS)
    if not args.include_lift:
        groups.pop("lift")
    if args.skip_head:
        groups.pop("head")

    rclpy.init()
    node = GoToInitialState()
    try:
        if not node.wait_joint_states():
            print("ERROR: no /joint_states received — is the follower bringup running?", file=sys.stderr)
            return 2
        current = node.joint_state

        # Build per-group trajectories and the move summary.
        plan = {}
        max_delta = 0.0
        print(f"{'joint':20s}{'current':>10s}{'target':>10s}{'delta':>10s}")
        for group, (topic, pred) in groups.items():
            names = [n for n in target if pred(n)]
            missing = [n for n in names if n not in current]
            if missing:
                print(f"ERROR: joints not in /joint_states: {missing}", file=sys.stderr)
                return 2
            for n in names:
                d = target[n] - current[n]
                max_delta = max(max_delta, abs(d))
                print(f"{n:20s}{current[n]:10.3f}{target[n]:10.3f}{d:10.3f}")
            plan[group] = (topic, names, [target[n] for n in names])
        print(f"\nmax |delta| = {max_delta:.3f} rad over {args.duration:.1f} s  "
              f"(peak ~{max_delta/args.duration:.2f} rad/s)")

        # One command source only: abort if anything else is ACTIVELY publishing on a
        # target topic. Publishers that send nothing for 1 s (stale Zenoh liveliness
        # tokens from a closed policy client, or a disarmed vr_controller) only warn,
        # unless --strict.
        busy = {t: node.count_publishers(t) for t, _, _ in plan.values()}
        busy = {t: c for t, c in busy.items() if c > 0}
        if busy:
            counts = {t: 0 for t in busy}
            subs = [node.create_subscription(JointTrajectory, t,
                                             lambda m, t=t: counts.__setitem__(t, counts[t] + 1), 10)
                    for t in busy]
            node.spin_for(1.0)
            for sub in subs:
                node.destroy_subscription(sub)
            live = {t: n for t, n in counts.items() if n > 0}
            idle = {t: busy[t] for t in busy if t not in live}
            if live:
                print("ERROR: other node(s) are actively publishing on target topics "
                      "(movej / leader / armed policy?):", file=sys.stderr)
                for t, n in live.items():
                    print(f"  {t}: {n} msgs in 1 s", file=sys.stderr)
                return 3
            if idle:
                print("WARNING: idle publisher(s) present on target topics "
                      "(stale tokens or a disarmed controller) — no traffic seen in 1 s:")
                for t, c in idle.items():
                    print(f"  {t}: {c} publisher(s)")
                if args.strict:
                    print("--strict given: aborting.", file=sys.stderr)
                    return 3
                print("Make sure no controller gets armed while the move runs.")

        if args.dry_run:
            print("dry-run: nothing published.")
            return 0
        if not args.yes:
            ans = input("Send this move to the robot? [y/N] ").strip().lower()
            if ans != "y":
                print("aborted.")
                return 1

        pubs = {}
        for group, (topic, names, positions) in plan.items():
            pubs[group] = node.create_publisher(JointTrajectory, topic, 10)
        node.spin_for(0.5)  # let publishers match subscribers over Zenoh

        for group, (topic, names, positions) in plan.items():
            msg = JointTrajectory()
            msg.joint_names = names
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.time_from_start.sec = int(math.floor(args.duration))
            pt.time_from_start.nanosec = int((args.duration - math.floor(args.duration)) * 1e9)
            msg.points.append(pt)
            pubs[group].publish(msg)
            print(f"published {group}: {len(names)} joints -> {topic}")

        node.spin_for(args.duration + 2.0)
        current = node.joint_state
        worst = 0.0
        print(f"\n{'joint':20s}{'now':>10s}{'target':>10s}{'error':>10s}")
        for group, (topic, names, positions) in plan.items():
            for n, p in zip(names, positions):
                e = current[n] - p
                worst = max(worst, abs(e))
                flag = "  <-- exceeds tolerance" if abs(e) > args.tolerance else ""
                print(f"{n:20s}{current[n]:10.3f}{p:10.3f}{e:10.3f}{flag}")
        print(f"\nworst |error| = {worst:.3f} rad (tolerance {args.tolerance})")
        return 0 if worst <= args.tolerance else 4
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
