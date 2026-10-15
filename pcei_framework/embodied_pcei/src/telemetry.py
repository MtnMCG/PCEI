"""
embodied_pcei/src/telemetry.py

Thread-safe telemetry management and scientific logger for Embodied PCEI.
Maintains live operational stats, rolling history for the dashboard,
and produces structured scientific logs in Markdown and JSONL.
"""

import time
import json
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from embodied_pcei.src.ninja_env import COURSE_WAYPOINTS

def get_waypoint_stage(x: float) -> str:
    if x < 9.0:
        return "Sprint Runway"
    elif x < 18.0:
        return "Quad Steps"
    elif x < 21.0:
        return "Platform 1"
    elif x < 28.0:
        return "Balance Beam"
    elif x < 30.0:
        return "Platform 2"
    elif x < 39.0:
        return "Hurdles"
    elif x < 46.0:
        return "Warped Wall"
    else:
        return "Summit Finish"

def get_waypoint_bonus(idx: int) -> int:
    if idx == len(COURSE_WAYPOINTS) - 1:
        return 100
    elif 3 <= idx <= 14:
        return 25
    else:
        return 15

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
JSONL_PATH = LOGS_DIR / "telemetry.jsonl"
SCIENTIFIC_LOG_PATH = LOGS_DIR / "SCIENTIFIC_LOG.md"


class TelemetryManager:
    def __init__(self, max_history: int = 500):
        self._lock = threading.Lock()
        self.max_history = max_history
        
        # Operational state
        self.status: str = "Initializing"
        self.start_time: float = time.time()
        self.iteration: int = 0
        self.global_step: int = 0
        self.total_episodes: int = 0
        self.fps: float = 0.0
        
        # Performance metrics
        self.mean_reward: float = 0.0
        self.max_reward: float = 0.0
        self.furthest_x: float = 0.0
        self.active_stage: str = "Stage 0: Sprint Runway"
        self.milestones_reached: int = 0
        
        # Realtime body kinematics & metabolism (from latest rendered agent)
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_z: float = 1.38
        self.current_vx: float = 0.0
        self.current_speed: float = 0.0
        self.stamina: float = 1.0
        self.health: float = 1.0
        self.hunger: float = 0.0
        
        # Optimization losses
        self.loss: float = 0.0
        self.policy_loss: float = 0.0
        self.value_loss: float = 0.0
        self.entropy: float = 0.0
        self.approx_kl: float = 0.0
        self.clip_frac: float = 0.0
        self.metabolic_loss: float = 0.0
        
        # Active camera & interactive controls
        self.active_camera: str = "behind_cam"
        self.ghost_swarm_enabled: bool = True
        self.ghost_swarm_alpha: float = 0.16
        self.perturbation_impulse: Optional[np.ndarray] = None
        
        # Evolutionary Darwinian tournament metrics
        self.generation: int = 0
        self.frontier_x: float = 4.2
        self.elite_fitness: float = 0.0
        self.mutations_count: int = 0
        self.view_mode: str = "alternate"  # "alternate", "pioneer", "anchor"
        
        # Progressive Course Navigation Waypoint state
        self.active_waypoint_idx: int = 0
        self.active_waypoint_name: str = "Runway Accelerate 1"
        self.dist_to_waypoint: float = 3.0

        # Flat Arena & Sequential Dynamic Random Goals state
        self.arena_mode: bool = True
        self.goals_completed: int = 0
        self.active_goal_pos: List[float] = [5.0, 0.0, 1.25]
        self.dist_to_goal: float = 5.0
        self.gaze_alignment: float = 0.0
        self.goal_arrival_radius: float = 2.0
        self.in_leeway: bool = False
        self.leeway_remaining: int = 0
        
        # 4-Stream Multi-Tasking Architecture State
        self.active_stream_view: str = "all"  # "all", "stand", "getup", "reach", "cruise"
        self.active_stream_mode: str = "cruise"
        self.active_reach_hand: str = "none"
        self.reach_touches: int = 0
        self.getup_success: bool = False
        self.stand_still_score: float = 0.0
        self.current_head_z: float = 0.30
        self.both_feet_down: bool = False
        self.one_foot_down: bool = False
        self.stream_stats: Dict[str, Dict[str, Any]] = {
            "stand": {"episodes": 0, "mean_reward": 0.0, "stillness_steps": 0, "mean_stillness": 0.0},
            "getup": {"episodes": 0, "mean_reward": 0.0, "successes": 0, "success_rate": 0.0},
            "reach": {"episodes": 0, "mean_reward": 0.0, "touches": 0, "mean_touches": 0.0},
            "cruise": {"episodes": 0, "mean_reward": 0.0, "goals": 0, "mean_goals": 0.0},
        }
        self.stream_generations: Dict[str, int] = {"stand": 0, "getup": 0, "reach": 0, "cruise": 0}
        self.stream_elite_fitness: Dict[str, float] = {"stand": 0.0, "getup": 0.0, "reach": 0.0, "cruise": 0.0}

        # Frame buffer for MJPEG streaming
        self.latest_frame_jpeg: Optional[bytes] = None
        self.frame_seq: int = 0
        self.latest_lidar_grid: Optional[List[List[float]]] = None
        
        # Rolling time series history for dashboard charts
        self.history_steps = deque(maxlen=max_history)
        self.history_rewards = deque(maxlen=max_history)
        self.history_furthest_x = deque(maxlen=max_history)
        self.history_loss = deque(maxlen=max_history)
        self.history_entropy = deque(maxlen=max_history)
        self.history_fps = deque(maxlen=max_history)
        self.history_stamina = deque(maxlen=max_history)

        # Milestone timestamps: {milestone_name: timestamp_str}
        self.milestone_log: List[Dict[str, Any]] = []

        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if not SCIENTIFIC_LOG_PATH.exists():
            self._init_scientific_log()

    def _init_scientific_log(self):
        header = """# PCEI Embodied Humanoid Locomotion & Ninja Warrior Course — Scientific Experiment Log

## Experiment Metadata
- **Project**: Embodied PCEI Physics Sandbox
- **Agent Architecture**: Dual-Timescale Recurrent PCEI (Mamba SSM + BitNet 1.58b Ternary Linear + FastKAN Non-linear Motor Synthesis + AmbientField Subconscious Field)
- **Physics Simulation Engine**: MuJoCo 3.14.0 (Native Apple Silicon Acceleration)
- **Actuation Topology**: 21 Degrees of Freedom (Torque Motor Controllable Joints)
- **Course Domain**: American Ninja Warrior Multi-Stage Obstacle Track (Runway -> Quad Steps -> Balance Beam -> Hurdles -> Warped Wall)
- **Optimization Strategy**: Actor-Critic Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
- **Start Timestamp**: {timestamp}

---

## Stage Progression Benchmarks
| Milestone | Course X (m) | Status | Discovery Iteration | Global Steps | Wall Clock Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 0: Runway Stability** | 0.0 - 8.0 m | In Progress | Iteration 0 | 0 | 0s |
| **Stage 1: Quad Steps** | 8.0 - 18.0 m | Locked | Pending | Pending | Pending |
| **Stage 2: Balance Beam** | 18.0 - 28.0 m | Locked | Pending | Pending | Pending |
| **Stage 3: Hurdles** | 28.0 - 38.0 m | Locked | Pending | Pending | Pending |
| **Stage 4: Warped Wall Ramp** | 38.0 - 46.0 m | Locked | Pending | Pending | Pending |
| **Course Summit Buzzer** | 46.0+ m | Locked | Pending | Pending | Pending |

---

## Chronological Scientific Telemetry & Analysis

""".format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        with open(SCIENTIFIC_LOG_PATH, "w") as f:
            f.write(header)

    def update_frame(self, jpeg_bytes: bytes, kinematics: Optional[Dict[str, float]] = None):
        with self._lock:
            self.latest_frame_jpeg = jpeg_bytes
            self.frame_seq += 1
            if kinematics:
                self.current_x = kinematics.get("x", self.current_x)
                self.current_y = kinematics.get("y", self.current_y)
                self.current_z = kinematics.get("z", self.current_z)
                self.current_vx = kinematics.get("vx", self.current_vx)
                self.current_speed = kinematics.get("speed", abs(self.current_vx))
                self.stamina = kinematics.get("stamina", self.stamina)
                self.health = kinematics.get("health", self.health)
                self.hunger = kinematics.get("hunger", self.hunger)
                if "waypoint_idx" in kinematics:
                    self.active_waypoint_idx = int(kinematics["waypoint_idx"])
                    self.active_waypoint_name = str(kinematics.get("waypoint_name", ""))
                    self.dist_to_waypoint = float(kinematics.get("dist_to_wp", 0.0))
                if "goals_completed" in kinematics:
                    self.goals_completed = int(kinematics["goals_completed"])
                if "dist_to_goal" in kinematics:
                    self.dist_to_goal = float(kinematics["dist_to_goal"])
                if "goal_arrival_radius" in kinematics:
                    self.goal_arrival_radius = float(kinematics["goal_arrival_radius"])
                if "goal_pos" in kinematics:
                    self.active_goal_pos = list(kinematics["goal_pos"])
                if "gaze_alignment" in kinematics:
                    self.gaze_alignment = float(kinematics["gaze_alignment"])
                if "in_leeway" in kinematics:
                    self.in_leeway = bool(kinematics["in_leeway"])
                if "leeway_remaining" in kinematics:
                    self.leeway_remaining = int(kinematics["leeway_remaining"])
                if "stream_mode" in kinematics:
                    self.active_stream_mode = str(kinematics["stream_mode"])
                if "active_reach_hand" in kinematics:
                    self.active_reach_hand = str(kinematics["active_reach_hand"])
                if "reach_touches" in kinematics:
                    self.reach_touches = int(kinematics["reach_touches"])
                if "getup_success" in kinematics:
                    self.getup_success = bool(kinematics["getup_success"])
                if "stand_still_score" in kinematics:
                    self.stand_still_score = float(kinematics["stand_still_score"])
                if "head_z" in kinematics:
                    self.current_head_z = float(kinematics["head_z"])
                if "both_feet_down" in kinematics:
                    self.both_feet_down = bool(kinematics["both_feet_down"])
                if "one_foot_down" in kinematics:
                    self.one_foot_down = bool(kinematics["one_foot_down"])

    def update_lidar(self, grid: Optional[List[List[float]]]):
        with self._lock:
            self.latest_lidar_grid = grid

    def update_iteration(self, iteration: int, global_step: int, fps: float, rollout_stats: Dict[str, Any], ppo_metrics: Dict[str, float], evolution_stats: Optional[Dict[str, Any]] = None):
        with self._lock:
            self.iteration = iteration
            self.global_step = global_step
            self.fps = fps
            self.status = "Training"
            
            self.mean_reward = rollout_stats.get("mean_reward", self.mean_reward)
            self.max_reward = rollout_stats.get("max_reward", self.max_reward)
            self.furthest_x = max(self.furthest_x, rollout_stats.get("furthest_x", 0.0))
            self.milestones_reached = rollout_stats.get("milestones", self.milestones_reached)
            
            if evolution_stats:
                self.generation = evolution_stats.get("generation", self.generation)
                self.frontier_x = evolution_stats.get("frontier_x", self.frontier_x)
                self.elite_fitness = evolution_stats.get("elite_fitness", self.elite_fitness)
                self.mutations_count = evolution_stats.get("total_mutations", self.mutations_count)
                if "stream_generations" in evolution_stats:
                    self.stream_generations = evolution_stats["stream_generations"]
                if "stream_elite_fitness" in evolution_stats:
                    self.stream_elite_fitness = evolution_stats["stream_elite_fitness"]
            
            if "stream_stats" in rollout_stats:
                self.stream_stats = rollout_stats["stream_stats"]
            
            # Compute stage name
            if self.arena_mode:
                if "goals_completed" in rollout_stats:
                    self.goals_completed = max(self.goals_completed, int(rollout_stats["goals_completed"]))
                self.active_stage = f"Flat Arena: {self.goals_completed} Goals Cleared (Target: {self.dist_to_goal:.1f}m)"
            else:
                if self.furthest_x < 8.0:
                    self.active_stage = f"Stage 0: Sprint Runway ({self.furthest_x:.1f}m / 8m)"
                elif self.furthest_x < 18.0:
                    self.active_stage = f"Stage 1: Quad Steps ({self.furthest_x:.1f}m / 18m)"
                elif self.furthest_x < 28.0:
                    self.active_stage = f"Stage 2: Balance Beam ({self.furthest_x:.1f}m / 28m)"
                elif self.furthest_x < 38.0:
                    self.active_stage = f"Stage 3: Hurdles ({self.furthest_x:.1f}m / 38m)"
                elif self.furthest_x < 46.0:
                    self.active_stage = f"Stage 4: Warped Wall ({self.furthest_x:.1f}m / 46m)"
                else:
                    self.active_stage = f"Course Complete: Summit Buzzer! ({self.furthest_x:.1f}m)"

            # Loss metrics
            self.loss = ppo_metrics.get("loss", self.loss)
            self.policy_loss = ppo_metrics.get("policy_loss", self.policy_loss)
            self.value_loss = ppo_metrics.get("value_loss", self.value_loss)
            self.entropy = ppo_metrics.get("entropy", self.entropy)
            self.approx_kl = ppo_metrics.get("approx_kl", self.approx_kl)
            self.clip_frac = ppo_metrics.get("clip_frac", self.clip_frac)
            self.metabolic_loss = ppo_metrics.get("metabolic_loss", self.metabolic_loss)

            # Append to rolling history
            self.history_steps.append(global_step)
            self.history_rewards.append(round(self.mean_reward, 2))
            self.history_furthest_x.append(round(self.furthest_x, 2))
            self.history_loss.append(round(self.loss, 4))
            self.history_entropy.append(round(self.entropy, 3))
            self.history_fps.append(round(fps, 1))
            self.history_stamina.append(round(self.stamina, 2))

        # Write structured telemetry line
        telemetry_entry = {
            "timestamp": time.time(),
            "iteration": iteration,
            "global_step": global_step,
            "fps": round(fps, 1),
            "generation": self.generation,
            "frontier_x": round(self.frontier_x, 2),
            "mutations": self.mutations_count,
            "mean_reward": round(self.mean_reward, 2),
            "max_reward": round(self.max_reward, 2),
            "furthest_x": round(self.furthest_x, 2),
            "stage": self.active_stage,
            "milestones": self.milestones_reached,
            "loss": round(self.loss, 4),
            "policy_loss": round(self.policy_loss, 4),
            "value_loss": round(self.value_loss, 4),
            "entropy": round(self.entropy, 3),
            "approx_kl": round(self.approx_kl, 5),
            "clip_frac": round(self.clip_frac, 3),
            "metabolic_loss": round(self.metabolic_loss, 4),
        }
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(telemetry_entry) + "\n")

        # Periodically log scientific summary markdown
        if iteration % 5 == 0 or iteration == 1:
            self._append_scientific_log(telemetry_entry)

    def _append_scientific_log(self, entry: Dict[str, Any]):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}h {mins:02d}m {secs:02d}s"

        md_block = f"""### Checkpoint Iteration {entry['iteration']} (Global Step: {entry['global_step']:,})
- **Wall Time Elapsed**: {time_str} | **Simulation Throughput**: {entry['fps']} SPS
- **Darwinian Generation**: `Gen {entry.get('generation', 0)}` | **Pioneer Frontier**: `{entry.get('frontier_x', 0.0):.2f}m` | **Codon Mutations**: `{entry.get('mutations', 0):,}`
- **Stage Progression**: `{entry['stage']}`
- **Furthest Penetration Distance**: `{entry['furthest_x']:.2f} meters`
- **Mean Episode Return**: `{entry['mean_reward']:.2f}` (Peak Return: `{entry['max_reward']:.2f}`)
- **PPO Optimization State**:
  * Total Loss: `{entry['loss']:.4f}` (Surrogate: `{entry['policy_loss']:.4f}` | Value: `{entry['value_loss']:.4f}`)
  * Policy Entropy: `{entry['entropy']:.3f}` | Approx KL: `{entry['approx_kl']:.5f}` | Clip Frac: `{entry['clip_frac']:.1%}`
- **Subconscious Homeostasis**: Stamina `{self.stamina:.1%}` | Health `{self.health:.1%}` | Hunger `{self.hunger:.1%}`

"""
        with open(SCIENTIFIC_LOG_PATH, "a") as f:
            f.write(md_block)

    def get_status_dict(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self.start_time
            return {
                "status": self.status,
                "elapsed_seconds": int(elapsed),
                "iteration": self.iteration,
                "global_step": self.global_step,
                "generation": self.generation,
                "frontier_x": round(self.frontier_x, 2),
                "elite_fitness": round(self.elite_fitness, 2),
                "mutations_count": self.mutations_count,
                "total_episodes": self.total_episodes,
                "fps": round(self.fps, 1),
                "mean_reward": round(self.mean_reward, 2),
                "max_reward": round(self.max_reward, 2),
                "furthest_x": round(self.furthest_x, 2),
                "active_stage": self.active_stage,
                "milestones": self.milestones_reached,
                "active_waypoint_idx": self.active_waypoint_idx,
                "active_waypoint_name": self.active_waypoint_name,
                "dist_to_waypoint": round(self.dist_to_waypoint, 2),
                "arena_mode": self.arena_mode,
                "goals_completed": self.goals_completed,
                "active_goal_pos": [round(float(c), 2) for c in self.active_goal_pos],
                "dist_to_goal": round(self.dist_to_goal, 2),
                "goal_arrival_radius": round(self.goal_arrival_radius, 2),
                "gaze_alignment": round(self.gaze_alignment, 3),
                "in_leeway": self.in_leeway,
                "leeway_remaining": self.leeway_remaining,
                "kinematics": {
                    "x": round(self.current_x, 2),
                    "y": round(self.current_y, 2),
                    "z": round(self.current_z, 2),
                    "vx": round(self.current_vx, 2),
                    "speed": round(self.current_speed, 2),
                },
                "metabolism": {
                    "stamina": round(self.stamina, 2),
                    "health": round(self.health, 2),
                    "hunger": round(self.hunger, 2),
                },
                "losses": {
                    "loss": round(self.loss, 4),
                    "policy_loss": round(self.policy_loss, 4),
                    "value_loss": round(self.value_loss, 4),
                    "entropy": round(self.entropy, 3),
                    "metabolic_loss": round(self.metabolic_loss, 4),
                },
                "active_stream_view": self.active_stream_view,
                "active_stream_mode": self.active_stream_mode,
                "active_reach_hand": self.active_reach_hand,
                "reach_touches": self.reach_touches,
                "getup_success": self.getup_success,
                "stand_still_score": round(self.stand_still_score, 2),
                "current_head_z": round(self.current_head_z, 2),
                "both_feet_down": self.both_feet_down,
                "one_foot_down": self.one_foot_down,
                "stream_stats": self.stream_stats,
                "stream_generations": self.stream_generations,
                "stream_elite_fitness": self.stream_elite_fitness,
                "camera": self.active_camera,
                "view_mode": self.view_mode,
                "ghost_swarm": self.ghost_swarm_enabled,
                "ghost_swarm_alpha": round(self.ghost_swarm_alpha, 2),
                "lidar": self.latest_lidar_grid,
                "waypoints": [
                    {
                        "index": idx,
                        "name": wp_name,
                        "x": round(float(coord[0]), 2),
                        "y": round(float(coord[1]), 2),
                        "z": round(float(coord[2]), 2),
                        "radius": round(float(radius), 2),
                        "stage": get_waypoint_stage(float(coord[0])),
                        "bonus": get_waypoint_bonus(idx),
                        "completed": idx < self.active_waypoint_idx,
                        "active": idx == self.active_waypoint_idx,
                        "dist": round(float(np.sqrt((self.current_x - coord[0])**2 + (self.current_y - coord[1])**2 + (self.current_z - coord[2])**2)), 2),
                    }
                    for idx, (coord, wp_name, radius) in enumerate(COURSE_WAYPOINTS)
                ],
            }

    def get_history_dict(self) -> Dict[str, List]:
        with self._lock:
            return {
                "steps": list(self.history_steps),
                "rewards": list(self.history_rewards),
                "furthest_x": list(self.history_furthest_x),
                "loss": list(self.history_loss),
                "entropy": list(self.history_entropy),
                "fps": list(self.history_fps),
                "stamina": list(self.history_stamina),
            }

    def get_latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_frame_jpeg

    def set_camera(self, camera_name: str):
        with self._lock:
            self.active_camera = camera_name

    def set_ghost_swarm(self, enabled: bool):
        with self._lock:
            self.ghost_swarm_enabled = bool(enabled)

    def set_ghost_swarm_alpha(self, alpha: float):
        with self._lock:
            self.ghost_swarm_alpha = max(0.02, min(1.0, float(alpha)))

    def toggle_ghost_swarm(self) -> bool:
        with self._lock:
            self.ghost_swarm_enabled = not self.ghost_swarm_enabled
            return self.ghost_swarm_enabled

    def set_view_mode(self, mode: str):
        with self._lock:
            if mode in ("alternate", "pioneer", "anchor"):
                self.view_mode = mode

    def set_stream_view(self, stream_mode: str):
        with self._lock:
            if stream_mode in ("all", "stand", "getup", "reach", "cruise"):
                self.active_stream_view = stream_mode

    def trigger_nudge(self, impulse_x: float, impulse_y: float):
        with self._lock:
            self.perturbation_impulse = np.array([impulse_x, impulse_y, 0.0], dtype=np.float32)

    def pop_nudge(self) -> Optional[np.ndarray]:
        with self._lock:
            imp = self.perturbation_impulse
            self.perturbation_impulse = None
            return imp


# Global singleton telemetry instance
GLOBAL_TELEMETRY = TelemetryManager()
