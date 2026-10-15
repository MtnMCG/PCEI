"""
embodied_pcei/src/ppo.py

Proximal Policy Optimization (PPO) training engine for Embodied PCEI.
Optimized for Apple Silicon (MPS) with Generalized Advantage Estimation (GAE),
clipped surrogate objective, entropy regularization, and subconscious AmbientField memory ticks.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from embodied_pcei.src.pcei_brain import PCEIActorCritic
from embodied_pcei.src.vector_env import VectorNinjaEnv
from src.modules.ternary import TernaryLinear


class PPOTrainer:
    def __init__(
        self,
        model: PCEIActorCritic,
        vec_env: VectorNinjaEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.005,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        metabolic_coef: float = 1e-4,
        n_steps: int = 512,
        n_epochs: int = 6,
        batch_size: int = 512,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.obs_dim = vec_env.obs_dim
        self.action_dim = vec_env.action_dim

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.metabolic_coef = metabolic_coef

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.total_batch_size = n_steps * self.num_envs

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # Rollout Storage Buffers
        self.obs_buf = torch.zeros((n_steps, self.num_envs, self.obs_dim), device=self.device)
        self.actions_buf = torch.zeros((n_steps, self.num_envs, self.action_dim), device=self.device)
        self.logprobs_buf = torch.zeros((n_steps, self.num_envs), device=self.device)
        self.rewards_buf = torch.zeros((n_steps, self.num_envs), device=self.device)
        self.dones_buf = torch.zeros((n_steps, self.num_envs), device=self.device)
        self.values_buf = torch.zeros((n_steps, self.num_envs), device=self.device)

        # Global counters
        self.global_step = 0
        self.total_episodes = 0

    def collect_rollouts(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Collects `n_steps` across `num_envs` parallel simulation environments.
        """
        self.model.eval()
        episode_returns = []
        episode_lengths = []
        furthest_xs = []
        milestone_counts = []
        goals_counts = []
        stream_episodes = {"stand": [], "getup": [], "reach": [], "cruise": []}

        with torch.no_grad():
            for step in range(self.n_steps):
                self.obs_buf[step] = next_obs
                self.dones_buf[step] = next_done

                # Sample actions from PCEI Actor-Critic
                action, log_prob, value, _ = self.model.get_action(next_obs)
                self.actions_buf[step] = action
                self.logprobs_buf[step] = log_prob
                self.values_buf[step] = value

                # Step environments on CPU / MuJoCo
                cpu_actions = action.cpu().numpy()
                step_obs, step_rewards, step_dones, infos = self.vec_env.step(cpu_actions)

                self.rewards_buf[step] = torch.tensor(step_rewards, device=self.device, dtype=torch.float32)
                next_obs = torch.tensor(step_obs, device=self.device, dtype=torch.float32)
                next_done = torch.tensor(step_dones, device=self.device, dtype=torch.float32)

                self.global_step += self.num_envs

                for info in infos:
                    if "episode" in info:
                        self.total_episodes += 1
                        ep = info["episode"]
                        episode_returns.append(ep["r"])
                        episode_lengths.append(ep["l"])
                        furthest_xs.append(ep["furthest_x"])
                        milestone_counts.append(ep["milestones"])
                        goals_counts.append(ep.get("goals_completed", 0))
                        sm = ep.get("stream_mode", "cruise")
                        if sm in stream_episodes:
                            stream_episodes[sm].append(ep)

        stream_stats = {}
        for sm in ("stand", "getup", "reach", "cruise"):
            eps = stream_episodes[sm]
            if eps:
                r_vals = [e["r"] for e in eps]
                mean_r = float(np.mean(r_vals))
                if sm == "stand":
                    still_steps = [e.get("stand_still_steps", 0) for e in eps]
                    stream_stats["stand"] = {
                        "episodes": len(eps),
                        "mean_reward": round(mean_r, 2),
                        "stillness_steps": int(sum(still_steps)),
                        "mean_stillness": round(float(np.mean(still_steps)), 2),
                    }
                elif sm == "getup":
                    succs = [1 if e.get("getup_success", False) else 0 for e in eps]
                    stream_stats["getup"] = {
                        "episodes": len(eps),
                        "mean_reward": round(mean_r, 2),
                        "successes": int(sum(succs)),
                        "success_rate": round(float(np.mean(succs)), 2),
                    }
                elif sm == "reach":
                    touches = [e.get("reach_touches", 0) for e in eps]
                    stream_stats["reach"] = {
                        "episodes": len(eps),
                        "mean_reward": round(mean_r, 2),
                        "touches": int(sum(touches)),
                        "mean_touches": round(float(np.mean(touches)), 2),
                    }
                elif sm == "cruise":
                    goals = [e.get("goals_completed", 0) for e in eps]
                    stream_stats["cruise"] = {
                        "episodes": len(eps),
                        "mean_reward": round(mean_r, 2),
                        "goals": int(sum(goals)),
                        "mean_goals": round(float(np.mean(goals)), 2),
                    }
            else:
                stream_stats[sm] = {
                    "episodes": 0,
                    "mean_reward": 0.0,
                }

        rollout_stats = {
            "mean_reward": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "max_reward": float(np.max(episode_returns)) if episode_returns else 0.0,
            "mean_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "furthest_x": float(np.max(furthest_xs)) if furthest_xs else 0.0,
            "milestones": int(np.max(milestone_counts)) if milestone_counts else 0,
            "goals_completed": int(np.max(goals_counts)) if goals_counts else 0,
            "episodes_collected": len(episode_returns),
            "stream_stats": stream_stats,
        }

        return next_obs, next_done, rollout_stats

    def compute_gae_and_returns(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) and bootstrap targets.
        """
        self.model.eval()
        with torch.no_grad():
            next_value = self.model.get_value(next_obs)
            advantages = torch.zeros_like(self.rewards_buf)
            lastgaelam = 0.0

            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones_buf[t + 1]
                    nextvalues = self.values_buf[t + 1]

                delta = self.rewards_buf[t] + self.gamma * nextvalues * nextnonterminal - self.values_buf[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.values_buf
            return advantages, returns

    def update_policy(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """
        Performs mini-batch PPO parameter updates with surrogate clipping.
        """
        self.model.train()

        # Flatten rollout dimensions
        b_obs = self.obs_buf.reshape(-1, self.obs_dim)
        b_actions = self.actions_buf.reshape(-1, self.action_dim)
        b_logprobs = self.logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values_buf.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_metabolic_loss = 0.0
        total_loss = 0.0
        approx_kl = 0.0
        clip_fracs = []

        batch_indices = np.arange(self.total_batch_size)
        num_updates = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, self.total_batch_size, self.batch_size):
                end = start + self.batch_size
                mb_inds = batch_indices[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_values = b_values[mb_inds]

                new_logprob, entropy, new_value = self.model.evaluate_actions(mb_obs, mb_actions)

                # Ratio: r(theta) = exp(log_prob - old_log_prob)
                log_ratio = new_logprob - mb_logprobs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # Diagnostic KL & clip fraction
                    kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kl += kl
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    clip_fracs.append(clip_frac)

                # Clipped surrogate policy loss
                surr1 = -mb_advantages * ratio
                surr2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pg_loss = torch.max(surr1, surr2).mean()

                # Clipped Value Function loss
                v_clipped = mb_values + torch.clamp(new_value - mb_values, -self.clip_eps, self.clip_eps)
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Policy entropy
                entropy_loss = -entropy.mean()

                # 4. Continuous Metabolic Decay & Sparsity Loss (Paper 07)
                # L2 metabolic decay on all network parameters
                metabolic_l2 = sum(p.pow(2).sum() for p in self.model.parameters())
                # L1 metabolic pruning pressure on TernaryLinear projection weights to restore 15-20% zero-state gates
                metabolic_sparsity = sum(m.weight.abs().sum() for m in self.model.modules() if isinstance(m, TernaryLinear))
                loss_metabolic = self.metabolic_coef * metabolic_l2 + (self.metabolic_coef * 0.5) * metabolic_sparsity

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss + loss_metabolic

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Re-sanitize actor exploration std to prevent bang-bang motor saturation
                if hasattr(self.model, "sanitize_log_std"):
                    self.model.sanitize_log_std()

                # Tick Ambient subconscious motor field with batch latents and loss
                with torch.no_grad():
                    features = self.model.extract_features(mb_obs[:16])
                    self.model.tick_ambient(features, loss.item())

                total_policy_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_metabolic_loss += loss_metabolic.item()
                total_loss += loss.item()
                num_updates += 1

        if num_updates > 0:
            return {
                "loss": total_loss / num_updates,
                "policy_loss": total_policy_loss / num_updates,
                "value_loss": total_value_loss / num_updates,
                "entropy": - (total_entropy_loss / num_updates),
                "approx_kl": approx_kl / num_updates,
                "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                "metabolic_loss": total_metabolic_loss / num_updates,
            }
        return {}

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "global_step": self.global_step,
            "total_episodes": self.total_episodes,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ambient_fast_state": self.model.ambient_field.fast_state,
            "ambient_slow_state": self.model.ambient_field.slow_state,
        }, path)

    def load_checkpoint(self, path: str, warm_start: bool = True, reset_legs: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        ckpt = torch.load(path, map_location=self.device)
        model_dict = ckpt["model_state_dict"]
        current_dict = self.model.state_dict()
        
        # Check if dimension surgery is needed (e.g. 76 -> 188 obs_dim)
        old_w = model_dict.get("obs_encoder.0.weight")
        current_w = current_dict.get("obs_encoder.0.weight")
        needs_surgery = (old_w is not None and current_w is not None and old_w.shape != current_w.shape)
        
        if warm_start and needs_surgery:
            print(f"[Embodied PCEI] Detected obs_dim expansion from {old_w.shape[1]} to {current_w.shape[1]}. Performing surgical weight transfer...")
            adapted_dict = {}
            for k, v in model_dict.items():
                if k == "obs_encoder.0.weight":
                    new_w = current_w.clone()
                    if old_w.shape[1] == 572 and current_w.shape[1] == 575:
                        # 1. Transfer core proprioception (57 features)
                        new_w[:, :57] = v[:, :57]
                        # 2. Transfer 512 LiDAR features
                        new_w[:, 57:569] = v[:, 57:569]
                        # 3. New Progressive Waypoint Guide (3 features): initialize with exact 0.0 (zero-shock invariant!)
                        new_w[:, 569:572] = 0.0
                        # 4. Transfer metabolic telemetry (last 3 features)
                        new_w[:, -3:] = v[:, -3:]
                    else:
                        # General fallback for arbitrary observation expansion
                        n_proprio = min(57, old_w.shape[1])
                        new_w[:, :n_proprio] = v[:, :n_proprio]
                        new_w[:, -3:] = v[:, -3:]
                        n_new_vision = new_w.shape[1] - n_proprio - 3
                        old_vision = v[:, 57:old_w.shape[1] - 3]
                        if old_vision.shape[1] > 0:
                            fan_in_scale = ((old_vision.shape[1] / max(1, n_new_vision)) ** 0.5) * 0.18
                            vision_mean = old_vision.mean(dim=-1, keepdim=True) * fan_in_scale
                            new_w[:, 57:-3] = vision_mean + torch.randn_like(new_w[:, 57:-3]) * (0.005 * fan_in_scale)
                    adapted_dict[k] = new_w
                elif k in current_dict and current_dict[k].shape == v.shape:
                    adapted_dict[k] = v
                else:
                    adapted_dict[k] = current_dict[k]
                    
            self.model.load_state_dict(adapted_dict, strict=True)
            # Re-initialize optimizer with new parameters
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
            print(f"[Embodied PCEI] Warm-start surgical transfer complete! Preserved 100% of Mamba SSM, BitNet, FastKAN, and Actor/Critic weights.")
        else:
            self.model.load_state_dict(model_dict)
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
                
        # Immediately sanitize actor_log_std so restored models do not suffer from bang-bang saturation
        if hasattr(self.model, "sanitize_log_std"):
            self.model.sanitize_log_std()
            print("[Embodied PCEI] Actor exploration std sanitized to continuous motor range [-3.0, -0.35] (std ~0.05-0.70)")

        # Surgically unfreeze arm actuator head (indices 15:21) if saturated
        with torch.no_grad():
            if hasattr(self.model, "actor_mean"):
                arm_weights = self.model.actor_mean[-1].weight[15:21]
                if arm_weights.abs().mean() > 0.10:
                    print(f"[Embodied PCEI] Detected saturated arm actuator weights (mean={arm_weights.abs().mean():.4f}). Performing surgical reset on arms (15:21)...")
                    cpu_w = torch.empty_like(arm_weights, device="cpu")
                    nn.init.orthogonal_(cpu_w, gain=0.01)
                    self.model.actor_mean[-1].weight[15:21].copy_(cpu_w.to(self.device))
                    self.model.actor_mean[-1].bias[15:21].zero_()
                    print("[Embodied PCEI] Arms unfrozen! Preserved 100% of core and leg locomotion weights (0:15).")

        # Surgically reset core and leg actuator head (indices 0:15) if requested,
        # purging prior 70M-step running bias and allowing clean Get-Up fall recovery learning
        if reset_legs:
            with torch.no_grad():
                if hasattr(self.model, "actor_mean"):
                    print("[Embodied PCEI] Performing surgical reset on core & legs (0:15) to eliminate running bias...")
                    cpu_w_legs = torch.empty_like(self.model.actor_mean[-1].weight[0:15], device="cpu")
                    nn.init.orthogonal_(cpu_w_legs, gain=0.01)
                    self.model.actor_mean[-1].weight[0:15].copy_(cpu_w_legs.to(self.device))
                    self.model.actor_mean[-1].bias[0:15].zero_()
                    self.model.actor_log_std[0:15].fill_(-0.6)
                    # Purge Adam momentum for fresh motor gradients
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
                    print("[Embodied PCEI] Core & leg reset complete! Running bias purged. 100% of Mamba SSM backbone, FastKAN, and Critic preserved.")

        self.global_step = ckpt.get("global_step", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)
        if "ambient_fast_state" in ckpt:
            self.model.ambient_field.fast_state = ckpt["ambient_fast_state"]
        if "ambient_slow_state" in ckpt:
            self.model.ambient_field.slow_state = ckpt["ambient_slow_state"]
