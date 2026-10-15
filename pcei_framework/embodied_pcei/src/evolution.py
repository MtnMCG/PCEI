"""
embodied_pcei/src/evolution.py

Darwinian Tournament & BitNet Ternary Codon Evolution Engine for Embodied PCEI.
Implements the Baldwin Effect:
- Intra-lifetime learning: Continuous policy gradients (PPO) optimize joint torques and reflexes.
- Inter-generational evolution: Darwinian selection, adaptive frontier colonization, and
  discrete ternary codon point mutations ({-1, 0, +1}) inject structural genetic novelty.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from embodied_pcei.src.vector_env import VectorNinjaEnv
from src.modules.ternary import TernaryLinear


class DarwinianTournament:
    """
    Orchestrates inter-generational evolutionary tournaments across parallel MuJoCo environments.
    """
    def __init__(
        self,
        vec_env: VectorNinjaEnv,
        model: nn.Module,
        elite_ratio: float = 0.25,
        mutation_rate: float = 0.0025,
    ):
        self.vec_env = vec_env
        self.model = model
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        
        self.generation = 0
        self.total_mutations = 0
        self.best_elite_fitness = -float("inf")
        self.best_elite_distance = 0.0

        # Stream-Specific Multi-Task Generational Tracking
        self.stream_generations = {"stand": 0, "getup": 0, "reach": 0, "cruise": 0}
        self.stream_elite_fitness = {"stand": 0.0, "getup": 0.0, "reach": 0.0, "cruise": 0.0}
        self.stream_records = {
            "stand": 0.0,   # Peak quiet stillness steps
            "getup": 0.0,   # Peak recovery success count
            "reach": 0.0,   # Peak target touches
            "cruise": 0.0,  # Peak goals cleared / furthest x
        }

    def compute_stream_fitness(self, ep: Dict[str, Any]) -> float:
        """
        Stream-specific Darwinian fitness function:
        - Stream 1 (Stand): Reference pose stillness steps + duration survival + efficiency.
        - Stream 2 (GetUp): Upright recovery jackpot + rapid recovery time bonus + efficiency.
        - Stream 3 (Reach): Bilateral target touches + spatial proximity + efficiency.
        - Stream 4 (Cruise): Goals reached + distance progressed + efficiency.
        """
        sm = ep.get("stream_mode", "cruise")
        ep_r = float(ep.get("r", 0.0))
        ep_l = max(1, int(ep.get("l", 1)))
        efficiency = ep_r / ep_l

        if sm == "stand":
            still_steps = float(ep.get("stand_still_steps", 0))
            survival_ratio = min(1.0, ep_l / 384.0)
            still_ratio = still_steps / max(1.0, float(ep_l))
            return (still_steps * 0.5) + (still_ratio * 50.0) + (survival_ratio * 50.0) + (0.3 * efficiency)
        elif sm == "getup":
            success = 100.0 if ep.get("getup_success", False) else 0.0
            speed_bonus = max(0.0, 150.0 - ep_l) * 0.5 if ep.get("getup_success", False) else 0.0
            return success + speed_bonus + (0.3 * efficiency)
        elif sm == "reach":
            touches = float(ep.get("reach_touches", 0))
            touch_cadence = (touches / ep_l) * 100.0  # touches per 100 steps (speed efficiency metric)
            return (touches * 40.0) + (touch_cadence * 30.0) + (0.5 * efficiency)
        else:  # cruise
            goals = float(ep.get("goals_completed", 0))
            x_dist = float(ep.get("furthest_x", 0.0))
            return (goals * 50.0) + x_dist + (0.3 * efficiency)

    def compute_episode_fitness(self, ep: Dict[str, Any]) -> float:
        """
        Multi-objective evolutionary fitness function:
        F = X_max + 0.3 * (Reward / Length)
        Reward per step acts as an efficiency and stability bonus.
        """
        x_dist = float(ep.get("furthest_x", 0.0))
        ep_r = float(ep.get("r", 0.0))
        ep_l = max(1, int(ep.get("l", 1)))
        efficiency = ep_r / ep_l
        return x_dist + (0.3 * efficiency)

    def audit_and_evolve(self, rollout_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a generational evolutionary audit:
        1. Ranks population fitness over recent completed episodes across all 4 streams.
        2. Computes stream-specific elite fitness and increments stream generations.
        3. Adapts the Pioneer Vanguard frontier waypoint based on trailblazer distance.
        4. Applies discrete BitNet ternary codon point mutations on actor-critic latent projections.
        """
        self.generation += 1
        recent_episodes = self.vec_env.completed_episodes[-64:]

        # ── MULTI-STREAM GENERATIONAL AUDIT ───────────────────────────────────
        stream_buckets: Dict[str, List[Dict[str, Any]]] = {"stand": [], "getup": [], "reach": [], "cruise": []}
        for ep in recent_episodes:
            sm = ep.get("stream_mode", "cruise")
            if sm in stream_buckets:
                stream_buckets[sm].append(ep)

        for sm, s_eps in stream_buckets.items():
            if s_eps:
                self.stream_generations[sm] += 1
                scored = [self.compute_stream_fitness(ep) for ep in s_eps]
                scored.sort(reverse=True)
                num_elites = max(1, int(len(scored) * self.elite_ratio))
                elite_fit = float(np.mean(scored[:num_elites]))
                self.stream_elite_fitness[sm] = round(elite_fit, 2)

                # Update stream-specific records
                if sm == "stand":
                    max_still = max(ep.get("stand_still_steps", 0) for ep in s_eps)
                    self.stream_records["stand"] = max(self.stream_records["stand"], float(max_still))
                elif sm == "getup":
                    successes = sum(1 for ep in s_eps if ep.get("getup_success", False))
                    self.stream_records["getup"] = max(self.stream_records["getup"], float(successes))
                elif sm == "reach":
                    max_touch = max(ep.get("reach_touches", 0) for ep in s_eps)
                    self.stream_records["reach"] = max(self.stream_records["reach"], float(max_touch))
                elif sm == "cruise":
                    max_g = max(ep.get("goals_completed", 0) for ep in s_eps)
                    self.stream_records["cruise"] = max(self.stream_records["cruise"], float(max_g))

        # Overall population trailblazer metrics (for frontier advancement & backwards compatibility)
        elite_fitness = 0.0
        max_dist_this_gen = 0.0

        if recent_episodes:
            scored_all = [(self.compute_episode_fitness(ep), ep.get("furthest_x", 0.0)) for ep in recent_episodes]
            scored_all.sort(key=lambda x: x[0], reverse=True)

            # Top quartile represents the elite trailblazers
            num_elites = max(1, int(len(scored_all) * self.elite_ratio))
            elite_fitness = float(np.mean([s[0] for s in scored_all[:num_elites]]))
            max_dist_this_gen = float(max([s[1] for s in scored_all]))
        elif rollout_stats:
            max_dist_this_gen = float(rollout_stats.get("furthest_x", 0.0))
            elite_fitness = max_dist_this_gen

        if elite_fitness > self.best_elite_fitness:
            self.best_elite_fitness = elite_fitness
        if max_dist_this_gen > self.best_elite_distance:
            self.best_elite_distance = max_dist_this_gen

        # ── 1. FRONTIER MILESTONE TRACKING ────────────────────────────────────
        # Tracks furthest continuous distance achieved by population from course origin
        current_frontier = self.vec_env.frontier_x
        if max_dist_this_gen > current_frontier:
            self.vec_env.advance_frontier(max_dist_this_gen)

        # ── 2. BITNET TERNARY CODON POINT MUTATIONS ───────────────────────────
        # In PCEI's BitNet 1.58b substrate, weights {-1, 0, +1} represent biological codons:
        #   +1: Excitatory agonist synapse
        #   -1: Inhibitory antagonist synapse
        #    0: Dormant non-coding intron
        # We apply discrete codon transitions to latent ternary projections.
        mutations_this_gen = 0
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, TernaryLinear):
                    W = m.weight.data
                    gamma = W.abs().mean().clamp(min=1e-4).item()
                    
                    # Stochastic mutation mask
                    rand_vals = torch.rand_like(W)
                    mutate_mask = rand_vals < self.mutation_rate
                    n_mut = int(mutate_mask.sum().item())
                    
                    if n_mut > 0:
                        mutations_this_gen += n_mut
                        
                        # Identify silent loci (|w| < 0.5*gamma) vs active loci
                        silent_mask = mutate_mask & (W.abs() < (0.5 * gamma))
                        active_mask = mutate_mask & (W.abs() >= (0.5 * gamma))
                        
                        # Silent -> Excitatory (+1) or Inhibitory (-1)
                        if silent_mask.any():
                            signs = torch.where(torch.rand_like(W) > 0.5, 1.0, -1.0)
                            W[silent_mask] = signs[silent_mask] * gamma * 1.1
                            
                        # Active -> Pruned / Silent (0)
                        if active_mask.any():
                            W[active_mask] = 0.0

        self.total_mutations += mutations_this_gen

        return {
            "generation": self.generation,
            "frontier_x": self.vec_env.frontier_x,
            "elite_fitness": round(elite_fitness, 2),
            "max_distance": round(max_dist_this_gen, 2),
            "mutations_this_gen": mutations_this_gen,
            "total_mutations": self.total_mutations,
            "stream_generations": dict(self.stream_generations),
            "stream_elite_fitness": dict(self.stream_elite_fitness),
            "stream_records": dict(self.stream_records),
        }
