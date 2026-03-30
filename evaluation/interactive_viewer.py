"""
Interactive Habitat-Sim viewer using OpenCV.
Navigate Matterport3D scenes with keyboard controls.

Controls:
    W/S     - Move forward/backward
    A/D     - Strafe left/right
    ←/→     - Turn left/right
    ↑/↓     - Look up/down
    Q       - Quit
    R       - Reset agent to start position
    P       - Print current agent position
    +/-     - Increase/decrease movement speed
"""

import argparse
import sys

import cv2
import numpy as np

import habitat_sim


def make_sim(scene_path, width=960, height=720):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    # RGB sensor
    rgb_spec = habitat_sim.SensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = np.array([height, width])
    rgb_spec.position = np.array([0.0, 1.5, 0.0])  # eye height 1.5m

    # Depth sensor
    depth_spec = habitat_sim.SensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = np.array([height, width])
    depth_spec.position = np.array([0.0, 1.5, 0.0])

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def depth_to_colormap(depth, max_depth=10.0):
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


def render_hud(frame, info_text):
    overlay = frame.copy()
    h = 30 + 22 * len(info_text)
    cv2.rectangle(overlay, (5, 5), (350, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(info_text):
        cv2.putText(frame, line, (10, 25 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Interactive Habitat-Sim Viewer")
    parser.add_argument("--scene", type=str,
                        default="data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
                        help="Path to scene .glb file")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--show-depth", action="store_true",
                        help="Show depth alongside RGB")
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}")
    sim = make_sim(args.scene, args.width, args.height)
    print("Scene loaded. Opening viewer window...")

    # Key mappings (OpenCV key codes)
    KEY_MAP = {
        ord("w"): "move_forward",
        ord("s"): "move_backward",
        ord("a"): "move_left",
        ord("d"): "move_right",
        82: "look_up",      # up arrow
        84: "look_down",    # down arrow
        81: "turn_left",    # left arrow
        83: "turn_right",   # right arrow
    }

    step_count = 0
    show_depth = args.show_depth
    window_name = "Habitat-Sim Interactive Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    obs = sim.get_sensor_observations()

    while True:
        # Render RGB
        rgb = obs["rgb"][:, :, :3]  # drop alpha
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Optionally show depth side-by-side
        if show_depth:
            depth_vis = depth_to_colormap(obs["depth"].squeeze())
            depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
            frame = np.hstack([frame, depth_vis])

        # HUD
        agent_state = sim.get_agent(0).get_state()
        pos = agent_state.position
        info = [
            "WASD=move  Arrows=look  Q=quit  R=reset",
            "T=toggle depth  P=print pos",
            f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            f"Steps: {step_count}",
        ]
        frame = render_hud(frame, info)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0) & 0xFF  # wait for keypress

        if key == ord("q") or key == 27:  # q or ESC
            break
        elif key == ord("r"):
            sim.reset()
            obs = sim.get_sensor_observations()
            step_count = 0
            continue
        elif key == ord("p"):
            state = sim.get_agent(0).get_state()
            print(f"Position: {state.position}")
            print(f"Rotation: {state.rotation}")
            continue
        elif key == ord("t"):
            show_depth = not show_depth
            continue
        elif key in KEY_MAP:
            action = KEY_MAP[key]
            obs = sim.step(action)
            step_count += 1
        else:
            # Unknown key, just refresh
            continue

    cv2.destroyAllWindows()
    sim.close()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
