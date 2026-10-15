"""
embodied_pcei/train.py

Master Training Runner for Embodied PCEI Ninja Warrior Locomotion.
Orchestrates:
- 8-way Parallel Vectorized MuJoCo Physics Environments
- PCEI Actor-Critic Network (Mamba SSM + BitNet Ternary + FastKAN + AmbientField) on Apple MPS
- Real-Time Live Visualizer Thread rendering high-FPS video for the Web Dashboard
- Embedded FastAPI / Uvicorn Web Server on Port 8080
- Continuous Checkpoint Management & Scientific Markdown / JSONL Telemetry Logging
"""

import os
import sys
import time
import io
import threading
import argparse
import numpy as np
import torch
import mujoco
from pathlib import Path
from PIL import Image

# Ensure framework root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from embodied_pcei.src.ninja_env import NinjaEnv
from embodied_pcei.src.vector_env import VectorNinjaEnv
from embodied_pcei.src.pcei_brain import PCEIActorCritic
from embodied_pcei.src.ppo import PPOTrainer
from embodied_pcei.src.evolution import DarwinianTournament
from embodied_pcei.src.telemetry import GLOBAL_TELEMETRY
from embodied_pcei.web.dashboard_server import start_server_in_thread

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
LATEST_CKPT = CHECKPOINTS_DIR / "pcei_ninja_latest.pt"
BEST_CKPT = CHECKPOINTS_DIR / "pcei_ninja_best.pt"


def visualizer_worker(
    model: PCEIActorCritic,
    device: torch.device,
    stop_event: threading.Event,
    num_ghosts: int = 7,
    arena_mode: bool = True,
):
    """
    Dedicated background worker that steps a physical evaluation cohort (1 primary runner + N ghost runners),
    renders camera frames at ~25 FPS with translucent ghost avatars, processes user perturbations,
    and updates the web dashboard's live MJPEG stream.
    """
    primary_env = NinjaEnv(frame_skip=5, arena_mode=arena_mode, stream_mode="cruise")
    ghost_envs = [NinjaEnv(frame_skip=5, arena_mode=arena_mode, stream_mode="cruise") for _ in range(num_ghosts)]

    ep_count = 0
    def get_eval_spawn(ep_num: int, is_ghost: bool = False) -> Tuple[float, float]:
        mode = GLOBAL_TELEMETRY.view_mode
        if mode == "pioneer" or (mode == "alternate" and ep_num % 2 == 1):
            return 0.0, 1.8
        return 0.0, 0.0

    def reset_eval_env(env: NinjaEnv, mode: str, ep_num: int, is_ghost: bool = False, ghost_idx: int = 0) -> np.ndarray:
        env.stream_mode = mode
        if mode == "stand":
            obs = env.reset(spawn_x=0.0, initial_vx=0.0)
            if is_ghost:
                # Distribute ghost standing avatars across lateral positions with subtle joint noise
                lane_offset = (ghost_idx + 1) * 0.40 * (-1 if ghost_idx % 2 == 0 else 1)
                env.data.qpos[1] = float(lane_offset)
                env.data.qpos[7:7 + env.n_actuators] += np.random.uniform(-0.015, 0.015, size=env.n_actuators)
                mujoco.mj_forward(env.model, env.data)
                obs = env._get_obs()
            return obs
        elif mode == "getup":
            if is_ghost:
                poses = ["prone", "supine", "lateral_left", "lateral_right", "kneeling_crumpled", "seated_slumped"]
                pose = poses[ghost_idx % len(poses)]
                obs = env.reset()
                env._apply_fall_state(pose)
                mujoco.mj_forward(env.model, env.data)
                return env._get_obs()
            return env.reset()
        elif mode == "reach":
            obs = env.reset(spawn_x=0.0, initial_vx=0.0)
            if is_ghost:
                env.active_reach_hand = "left" if (ghost_idx % 2 == 1) else "right"
                env._sample_reach_target()
                obs = env._get_obs()
            return obs
        else:
            p_sx, p_vx = get_eval_spawn(ep_num, is_ghost)
            obs = env.reset(spawn_x=p_sx, initial_vx=p_vx)
            if is_ghost:
                # Distribute ghost runners across parallel lanes so avatars don't stack on top of each other
                lane_offset = (ghost_idx + 1) * 0.45 * (-1 if ghost_idx % 2 == 0 else 1)
                env.data.qpos[1] = float(lane_offset)
                mujoco.mj_forward(env.model, env.data)
                obs = env._get_obs()
            return obs

    primary_obs = reset_eval_env(primary_env, "stand", ep_count)
    ghost_obses = [reset_eval_env(ge, "stand", ep_count, is_ghost=True, ghost_idx=i) for i, ge in enumerate(ghost_envs)]

    while not stop_event.is_set():
        cam_name = GLOBAL_TELEMETRY.active_camera
        swarm_on = GLOBAL_TELEMETRY.ghost_swarm_enabled

        # Stream switching check
        desired_view = GLOBAL_TELEMETRY.active_stream_view
        if desired_view != "all":
            target_mode = desired_view
        else:
            STREAMS = ["stand", "getup", "reach", "cruise"]
            target_mode = STREAMS[(ep_count // 2) % len(STREAMS)]

        if primary_env.stream_mode != target_mode:
            primary_obs = reset_eval_env(primary_env, target_mode, ep_count)
            for i, ge in enumerate(ghost_envs):
                ghost_obses[i] = reset_eval_env(ge, target_mode, ep_count, is_ghost=True, ghost_idx=i)

        # Check for user perturbation from the dashboard
        impulse = GLOBAL_TELEMETRY.pop_nudge()
        if impulse is not None:
            # Apply impulse force to torso qvel of primary agent
            primary_env.data.qvel[0] += impulse[0] * 0.1  # X push
            primary_env.data.qvel[1] += impulse[1] * 0.1  # Y lateral push

        # Batched inference for primary + ghost agents
        if swarm_on and num_ghosts > 0:
            all_obs = np.stack([primary_obs] + ghost_obses, axis=0)
            obs_tensor = torch.tensor(all_obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                actions, _, _, _ = model.get_action(obs_tensor, deterministic=False)
                actions_np = actions.cpu().numpy()
            primary_action = actions_np[0]
            ghost_actions = actions_np[1:]
        else:
            obs_tensor = torch.tensor(primary_obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_tensor, deterministic=False)
                primary_action = action.squeeze(0).cpu().numpy()
            ghost_actions = []

        # Step primary agent
        primary_obs, reward, terminated, truncated, info = primary_env.step(primary_action)
        if terminated or truncated:
            ep_count += 1
            cur_view = GLOBAL_TELEMETRY.active_stream_view
            next_mode = cur_view if cur_view != "all" else ["stand", "getup", "reach", "cruise"][(ep_count // 2) % 4]
            primary_obs = reset_eval_env(primary_env, next_mode, ep_count)
            if swarm_on and num_ghosts > 0:
                for i, ge in enumerate(ghost_envs):
                    ghost_obses[i] = reset_eval_env(ge, next_mode, ep_count, is_ghost=True, ghost_idx=i)

        # Step ghost agents independently
        ghost_datas = []
        if swarm_on and len(ghost_actions) > 0:
            for i, ge in enumerate(ghost_envs):
                g_obs, _, g_term, g_trunc, _ = ge.step(ghost_actions[i])
                if g_term or g_trunc:
                    cur_view = GLOBAL_TELEMETRY.active_stream_view
                    next_mode = cur_view if cur_view != "all" else ["stand", "getup", "reach", "cruise"][(ep_count // 2) % 4]
                    ghost_obses[i] = reset_eval_env(ge, next_mode, ep_count, is_ghost=True, ghost_idx=i)
                else:
                    ghost_obses[i] = g_obs
                ghost_datas.append(ge.data)

        # Render frame
        try:
            render_cam = cam_name
            if primary_env.stream_mode == "reach" and cam_name == "behind_cam":
                render_cam = "reach_cam"

            frame_rgb = primary_env.render_frame(
                width=480,
                height=320,
                camera_name=render_cam,
                ghost_datas=ghost_datas if swarm_on else None,
                ghost_alpha=GLOBAL_TELEMETRY.ghost_swarm_alpha,
            )
            img = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            jpeg_bytes = buf.getvalue()

            kinematics = {
                "x": info["x_pos"],
                "y": info["y_pos"],
                "z": info["z_pos"],
                "vx": info["vx"],
                "speed": info.get("torso_speed", abs(info["vx"])),
                "stamina": info["stamina"],
                "health": info["health"],
                "hunger": info["hunger"],
                "stream_mode": info.get("stream_mode", primary_env.stream_mode),
                "active_reach_hand": info.get("active_reach_hand", "none"),
                "reach_touches": info.get("reach_touches", 0),
                "getup_success": info.get("getup_success", False),
                "stand_still_score": info.get("stand_still_score", 0.0),
                "waypoint_idx": info.get("waypoint_idx", 0),
                "waypoint_name": info.get("waypoint_name", ""),
                "dist_to_wp": info.get("dist_to_wp", 0.0),
                "goals_completed": info.get("goals_completed", 0),
                "dist_to_goal": info.get("dist_to_goal", 0.0),
                "goal_arrival_radius": info.get("goal_arrival_radius", 2.0),
                "goal_pos": info.get("goal_pos", [5.0, 0.0, 1.25]),
                "target_local_pos": info.get("target_local_pos", [0.0, 0.0, 0.0]),
                "gaze_alignment": info.get("gaze_alignment", 0.0),
                "in_leeway": info.get("in_leeway", False),
                "leeway_remaining": info.get("leeway_remaining", 0),
            }
            GLOBAL_TELEMETRY.update_frame(jpeg_bytes, kinematics)
            if "lidar_grid" in info:
                GLOBAL_TELEMETRY.update_lidar(info["lidar_grid"])
        except Exception as e:
            pass

        time.sleep(0.04)  # Target ~25 FPS


def main():
    parser = argparse.ArgumentParser(description="Embodied PCEI Ninja Warrior Training")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel MuJoCo environments")
    parser.add_argument("--n_steps", type=int, default=512, help="Rollout steps per environment")
    parser.add_argument("--batch_size", type=int, default=512, help="PPO mini-batch size")
    parser.add_argument("--n_epochs", type=int, default=6, help="PPO update epochs per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--port", type=int, default=8080, help="Web dashboard port")
    parser.add_argument("--d_model", type=int, default=128, help="PCEI latent dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="PCEI diffusion layers")
    parser.add_argument("--frontier_x", type=float, default=0.0, help="Initial pioneer vanguard spawn distance")
    parser.add_argument("--num_ghosts", type=int, default=7, help="Number of translucent ghost avatars in live stream")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from latest checkpoint if available")
    parser.add_argument("--reset_legs", action="store_true", default=False, help="Surgically reset core and leg actor weights (0:15) to eliminate running bias and accelerate Get-Up recovery")
    parser.add_argument("--arena", action="store_true", default=True, help="Train in 50x50m flat arena with dynamic random goals (default: True)")
    parser.add_argument("--course", action="store_false", dest="arena", help="Train on linear American Ninja Warrior obstacle course")
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Embodied PCEI] Acceleration: Apple Silicon MPS Active")
    else:
        device = torch.device("cpu")
        print("[Embodied PCEI] Acceleration: CPU Fallback")

    # 2. Vectorized Environments (All spawn at X=0: 70% dead-stop, 30% sprint momentum)
    env_mode_str = "Flat Arena (Sequential Dynamic Goals)" if args.arena else "American Ninja Warrior Obstacle Course"
    print(f"[Embodied PCEI] Environment Mode: {env_mode_str}")
    print(f"[Embodied PCEI] Initializing {args.num_envs} Parallel MuJoCo Environments...")
    GLOBAL_TELEMETRY.arena_mode = args.arena
    vec_env = VectorNinjaEnv(num_envs=args.num_envs, frame_skip=5, frontier_x=args.frontier_x, arena_mode=args.arena)
    print(f"[Embodied PCEI] Observation Dim: {vec_env.obs_dim} | Action Dim (Actuators): {vec_env.action_dim}")
    print(f"[Embodied PCEI] Population Partition: {vec_env.anchor_count} Dead-Stop Anchors (X=0.0m) | {vec_env.pioneer_count} Momentum Runners (X=0.0m, Vx=1.8m/s)")
    GLOBAL_TELEMETRY.frontier_x = vec_env.frontier_x

    # 3. PCEI Neural Brain Policy & Value Network
    print(f"[Embodied PCEI] Constructing Dual-Timescale PCEI Brain (d_model={args.d_model}, layers={args.num_layers})...")
    model = PCEIActorCritic(
        obs_dim=vec_env.obs_dim,
        action_dim=vec_env.action_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Embodied PCEI] Model Parameter Count: {total_params:,}")

    # 4. Darwinian Evolutionary Tournament Engine
    tournament = DarwinianTournament(
        vec_env=vec_env,
        model=model,
        elite_ratio=0.25,
        mutation_rate=0.0025,
    )

    # 5. PPO Trainer
    trainer = PPOTrainer(
        model=model,
        vec_env=vec_env,
        lr=args.lr,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # Checkpoint restore
    if args.resume and LATEST_CKPT.exists():
        print(f"[Embodied PCEI] Found existing checkpoint at {LATEST_CKPT}. Restoring weights...")
        try:
            trainer.load_checkpoint(str(LATEST_CKPT), warm_start=True, reset_legs=args.reset_legs)
            print(f"[Embodied PCEI] Successfully restored at Global Step: {trainer.global_step:,}, Total Episodes: {trainer.total_episodes:,}")
        except Exception as e:
            print(f"[Embodied PCEI] Warning: Could not restore checkpoint: {e}. Starting fresh.")

    # 5. Start Web Dashboard Server
    print(f"[Embodied PCEI] Launching Real-Time Web Dashboard on http://localhost:{args.port}...")
    server_thread = start_server_in_thread(host="0.0.0.0", port=args.port)

    # 6. Start Live Visualizer Thread
    stop_visualizer = threading.Event()
    viz_thread = threading.Thread(
        target=visualizer_worker,
        args=(model, device, stop_visualizer, args.num_ghosts, args.arena),
        daemon=True,
    )
    viz_thread.start()
    print("[Embodied PCEI] Live Simulation Visualizer Thread Active.")

    # 7. Training Loop
    print("\n" + "="*80)
    print("  EMBODIED PCEI — HUMANOID NINJA WARRIOR REINFORCEMENT LEARNING RUNNER")
    print("="*80 + "\n")

    next_obs = torch.tensor(vec_env.reset(), device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    iteration = 0
    best_reward = -float("inf")
    best_distance = 0.0

    t_start = time.time()

    try:
        while True:
            iteration += 1
            iter_t0 = time.time()

            # Step 1: Collect parallel rollouts
            next_obs, next_done, rollout_stats = trainer.collect_rollouts(next_obs, next_done)

            # Step 2: Compute GAE and returns
            advantages, returns = trainer.compute_gae_and_returns(next_obs, next_done)

            # Step 3: PPO policy and value update
            ppo_metrics = trainer.update_policy(advantages, returns)

            iter_t1 = time.time()
            iter_duration = max(0.001, iter_t1 - iter_t0)
            fps = trainer.total_batch_size / iter_duration

            # Step 4: Inter-Generational Darwinian Tournament & BitNet Codon Evolution
            evolution_stats = None
            if iteration % 10 == 0:
                evolution_stats = tournament.audit_and_evolve(rollout_stats)
                print(
                    f"  >>> [Darwinian Gen {evolution_stats['generation']}] "
                    f"Pioneer Frontier: {evolution_stats['frontier_x']:.2f}m | "
                    f"Elite Fitness: {evolution_stats['elite_fitness']:.2f} | "
                    f"Codon Mutations: {evolution_stats['mutations_this_gen']} "
                    f"(Total: {evolution_stats['total_mutations']:,})\n"
                    f"      Streams -> Stand Gen {tournament.stream_generations['stand']} ({tournament.stream_elite_fitness['stand']:.1f}) | "
                    f"GetUp Gen {tournament.stream_generations['getup']} ({tournament.stream_elite_fitness['getup']:.1f}) | "
                    f"Reach Gen {tournament.stream_generations['reach']} ({tournament.stream_elite_fitness['reach']:.1f}) | "
                    f"Cruise Gen {tournament.stream_generations['cruise']} ({tournament.stream_elite_fitness['cruise']:.1f})"
                )

            # Update live telemetry
            GLOBAL_TELEMETRY.frontier_x = vec_env.frontier_x
            GLOBAL_TELEMETRY.update_iteration(
                iteration=iteration,
                global_step=trainer.global_step,
                fps=fps,
                rollout_stats=rollout_stats,
                ppo_metrics=ppo_metrics,
                evolution_stats=evolution_stats,
            )

            # Console logging
            cur_mean_r = rollout_stats["mean_reward"]
            cur_dist = rollout_stats["furthest_x"]
            cur_loss = ppo_metrics.get("loss", 0.0)
            cur_ent = ppo_metrics.get("entropy", 0.0)
            stage_name = GLOBAL_TELEMETRY.active_stage

            print(
                f"[Iter {iteration:4d}] "
                f"Step: {trainer.global_step:7,d} | "
                f"FPS: {fps:5.1f} | "
                f"Mean Return: {cur_mean_r:7.2f} | "
                f"Record: {GLOBAL_TELEMETRY.furthest_x:5.2f}m | "
                f"Frontier: {vec_env.frontier_x:4.1f}m | "
                f"Gen: {tournament.generation:2d} | "
                f"Loss: {cur_loss:6.3f} | "
                f"Ent: {cur_ent:5.2f}"
            )

            # Checkpoint saving
            if iteration % 5 == 0:
                trainer.save_checkpoint(str(LATEST_CKPT))

            if cur_dist > best_distance or cur_mean_r > best_reward:
                if cur_dist > best_distance:
                    best_distance = cur_dist
                if cur_mean_r > best_reward:
                    best_reward = cur_mean_r
                trainer.save_checkpoint(str(BEST_CKPT))
                print(f"  >>> New Record Achieved! (Distance: {best_distance:.2f}m, Return: {best_reward:.2f}) -> Saved {BEST_CKPT.name}")

    except KeyboardInterrupt:
        print("\n[Embodied PCEI] Training interrupted by user. Saving final checkpoint...")
        trainer.save_checkpoint(str(LATEST_CKPT))
        stop_visualizer.set()
        print("[Embodied PCEI] Checkpoint saved. Goodbye!")


if __name__ == "__main__":
    main()
