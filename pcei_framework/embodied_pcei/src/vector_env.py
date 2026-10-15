"""
embodied_pcei/src/vector_env.py

Synchronous Vectorized Environment for high-throughput parallel humanoid simulation.
Runs multiple MuJoCo instances seamlessly with zero IPC overhead.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from embodied_pcei.src.ninja_env import NinjaEnv


class VectorNinjaEnv:
    """
    Vectorized environment holding `num_envs` independent NinjaEnv instances.
    Provides batch step and reset operations with auto-reset upon termination.
    """
    def __init__(self, num_envs: int = 8, frame_skip: int = 5, frontier_x: float = 4.2, arena_mode: bool = True):
        self.num_envs = num_envs
        self.arena_mode = arena_mode
        
        # Partition num_envs across 4 Balanced Motor Learning Streams (25% each):
        # Stream 1: Quiet Standing Balance ('stand')
        # Stream 2: Fall Recovery ('getup')
        # Stream 3: Bilateral Hand Reaching ('reach')
        # Stream 4: Flat Arena Waypoint Cruising ('cruise')
        # For 16 envs: 4 Stand (25%), 4 Get-Up (25%), 4 Reach (25%), 4 Cruise (25%)
        # For 8 envs:  2 Stand, 2 Get-Up, 2 Reach, 2 Cruise
        n_per_stream = num_envs // 4
        rem = num_envs % 4
        counts = [n_per_stream + (1 if k < rem else 0) for k in range(4)]
        n_stand, n_getup, n_reach, n_cruise = counts

        self.stream_assignments = []
        for i in range(num_envs):
            if i < n_stand:
                self.stream_assignments.append("stand")
            elif i < n_stand + n_getup:
                self.stream_assignments.append("getup")
            elif i < n_stand + n_getup + n_reach:
                self.stream_assignments.append("reach")
            else:
                self.stream_assignments.append("cruise")

        self.envs = [
            NinjaEnv(frame_skip=frame_skip, arena_mode=arena_mode, stream_mode=self.stream_assignments[i])
            for i in range(num_envs)
        ]
        
        self.obs_dim = self.envs[0].observation_dim
        self.action_dim = self.envs[0].action_dim
        
        self.anchor_count = sum(1 for s in self.stream_assignments if s in ("stand", "getup", "reach"))
        self.pioneer_count = sum(1 for s in self.stream_assignments if s == "cruise")
        self.frontier_x = 0.0
        
        # Episode statistics tracking across envs
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.completed_episodes = []

    def _reset_env(self, env_idx: int, seed: Optional[int] = None) -> np.ndarray:
        """Resets individual environment according to its designated stream mode."""
        env = self.envs[env_idx]
        mode = self.stream_assignments[env_idx]
        if mode == "stand":
            return env.reset(seed=seed, spawn_x=0.0, initial_vx=0.0)
        elif mode == "getup":
            return env.reset(seed=seed)
        elif mode == "reach":
            return env.reset(seed=seed, spawn_x=0.0, initial_vx=0.0)
        else:
            # cruise mode: alternate dead-stop and momentum starts
            vx = 0.0 if (env_idx % 2 == 0) else 1.8
            return env.reset(seed=seed, spawn_x=0.0, initial_vx=vx)

    def advance_frontier(self, new_frontier: float):
        """Tracks furthest milestone progress achieved from the start line."""
        self.frontier_x = max(self.frontier_x, float(new_frontier))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        obs_list = []
        for i in range(self.num_envs):
            s = (seed + i) if seed is not None else None
            obs = self._reset_env(i, seed=s)
            obs_list.append(obs)
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
            
        return np.stack(obs_list, axis=0)  # (num_envs, obs_dim)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments with given actions.
        actions: (num_envs, action_dim)
        Returns:
            next_obs: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: list of info dicts
        """
        next_obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos = []

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(actions[i])
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            done = term or trunc
            dones[i] = done
            rewards[i] = reward
            
            if done:
                # Store final episode stats in info
                info["episode"] = {
                    "r": float(self.episode_rewards[i]),
                    "l": int(self.episode_lengths[i]),
                    "stream_mode": self.stream_assignments[i],
                    "furthest_x": float(env.furthest_x),
                    "milestones": len(env.milestones_reached),
                    "goals_completed": int(getattr(env, "goals_completed", 0)),
                    "reach_touches": int(getattr(env, "reach_touches", 0)),
                    "getup_success": bool(getattr(env, "getup_success_count", 0) > 0),
                    "stand_still_steps": int(getattr(env, "stand_still_steps", 0)),
                }
                info["terminal_observation"] = obs.copy()
                self.completed_episodes.append(info["episode"])
                
                # Auto-reset environment for continuous training flow
                obs = self._reset_env(i)
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                
            next_obs_list.append(obs)
            infos.append(info)

        return np.stack(next_obs_list, axis=0), rewards, dones, infos

    def render_frame(
        self,
        env_idx: int = 0,
        width: int = 480,
        height: int = 320,
        camera_name: str = "track_cam",
        ghost_datas: Optional[List[Any]] = None,
        ghost_alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Renders camera frame from designated environment index."""
        target_env = self.envs[min(max(0, env_idx), self.num_envs - 1)]
        return target_env.render_frame(
            width=width,
            height=height,
            camera_name=camera_name,
            ghost_datas=ghost_datas,
            ghost_alpha=ghost_alpha,
        )
