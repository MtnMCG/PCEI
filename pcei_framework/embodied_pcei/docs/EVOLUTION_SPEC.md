# Embodied PCEI: Generational Genetic Evolution & Mutation Specification

## 1. Executive Summary & Biological Foundation

The **Phase-Change Electro-Inductive (PCEI)** neural architecture bridges continuous physics-driven motor control with discrete digital genetics. 

In traditional deep neural networks, continuous 32-bit floating-point weights make true genetic algorithms unwieldy: mutating an arbitrary IEEE 754 float by adding Gaussian noise is biologically ungrounded and tends to shatter fragile internal representations. 

In contrast, **PCEI's native BitNet 1.58b ternary substrate ($\mathbf{W} \in \{-1, 0, +1\}$)** provides an organic digital analogue to **biological DNA codons**:
- $\mathbf{+1}$ = Excitatory synaptic pathway (Gene Expressed: Agonist)
- $\mathbf{-1}$ = Inhibitory synaptic pathway (Gene Expressed: Antagonist)
- $\mathbf{0}$ = Silent / Pruned non-coding locus (Intron / Inactive)

By combining **intra-lifetime reinforcement learning (PPO)** with **inter-generational Darwinian genetics (Crossover & Mutation)**, Embodied PCEI models the **Baldwin Effect**: traits that are initially learned through intensive lifetime trial-and-error become progressively hardcoded into innate genomic instincts across successive generations.

---

## 2. Genomic Architecture

Every Embodied PCEI organism possesses a dual-layer genome comprising:
1. **The Synaptic Chromosome ($\mathcal{G}_{\text{syn}}$)**: Discrete ternary weight matrices governing the Mamba SSM recurrences and Actor-Critic heads.
2. **The Morphological & Epigenetic Chromosome ($\mathcal{G}_{\text{epi}}$)**: Continuous hyperparameters determining FastKAN non-linear spline knots, subconscious field decay rates, and metabolic constants.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   EMBODIED PCEI DIPLOID GENOME                        │
├────────────────────────────────────────────────────────────────────────┤
│  [Chromosome 1: Locomotion Core]                                       │
│   ├── BitNet Ternary Weights W_ssm  ∈ {-1, 0, +1}^(128x128)            │
│   └── FastKAN Spline Knots B_motor  ∈ R^(64x21)                        │
│                                                                        │
│  [Chromosome 2: Homeostatic Metabolism]                                │
│   ├── Lung Stamina Decay Rate: λ_stamina ∈ [0.001, 0.05]               │
│   ├── Stomach Capacity: V_max ∈ [0.8, 1.5]                             │
│   └── Recovery Quotient: η_health ∈ [0.01, 0.1]                        │
│                                                                        │
│  [Chromosome 3: Subconscious Ambient Field]                            │
│   ├── AmbientField Spatial Dim: D_ambient = 16                         │
│   └── Temporal Smoothing Factor: τ_subconscious ∈ [0.85, 0.99]         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Genetic Operators

### 3.1 Point Mutations (Ternary Codon Flipping)

During gametogenesis / reproduction, each synaptic locus $w_{ij} \in \{-1, 0, +1\}$ is subjected to stochastic mutation governed by transition probability matrix $\mathbf{M}$:

$$\mathbf{M} = \begin{pmatrix} 
1 - \mu_{\text{decay}} & \mu_{\text{decay}} & 0 \\
\mu_{\text{act}} & 1 - 2\mu_{\text{act}} & \mu_{\text{act}} \\
0 & \mu_{\text{decay}} & 1 - \mu_{\text{decay}}
\end{pmatrix}$$

Where:
- $\mu_{\text{act}} = 0.002$ (Probability of activating a silent/dormant synaptic connection $0 \to +1$ or $0 \to -1$).
- $\mu_{\text{decay}} = 0.005$ (Probability of pruning an active connection back to neutral $1 \to 0$ or $-1 \to 0$, enforcing energetic parsimony).
- Direct polarity inversion ($+1 \leftrightarrow -1$) is biologically forbidden in a single step; it requires passing through the dormant intermediate state $0$.

### 3.2 Chromosomal Crossover (Synaptic Recombination)

When two parent organisms $P_A$ and $P_B$ reproduce, their offspring $O$ inherits a mosaic genome generated via **Functional Module Crossover**:

1. **Layer-Level Swapping**: High-level reflex modules (e.g. Quad-step balance stabilization vs. Sprint forward drive) are conserved as coherent structural blocks.
2. **Uniform Synaptic Crossover**: For shared latent projections:
   $$w_{ij}^{(O)} = \begin{cases} 
   w_{ij}^{(A)} & \text{with probability } 0.5 \\
   w_{ij}^{(B)} & \text{with probability } 0.5 
   \end{cases}$$
3. **Epigenetic Blending**: Continuous metabolic traits undergo arithmetic blending with random perturbation:
   $$\lambda^{(O)} = \alpha \lambda^{(A)} + (1 - \alpha) \lambda^{(B)} + \mathcal{N}(0, \sigma_{\text{mutation}}^2)$$
   where $\alpha \sim \text{Uniform}(0, 1)$.

---

## 4. Dual-Timescale Evolution: Lamarckian vs. Darwinian Dynamics

Embodied PCEI implements a hybrid evolutionary dynamic:

### Scale 1: Lifetime Learning (Intra-Generational PPO)
- Agents encounter physical obstacles (hurdles, quad steps, balance beams).
- Policy gradients backpropagate through the continuous surrogate networks to optimize locomotion rewards.
- The agent adapts its motor timing and stance articulation in real time.

### Scale 2: Evolutionary Selection (Inter-Generational Darwinism)
- At the end of each generation (or when an agent achieves a course milestone), individuals are evaluated on a **Multi-Objective Fitness Function**:
  $$\mathcal{F} = \beta_1 X_{\text{max}} + \beta_2 \bar{v}_x - \beta_3 E_{\text{torque}} + \beta_4 H_{\text{health}}$$
  * $X_{\text{max}}$: Furthest course penetration distance (meters)
  * $\bar{v}_x$: Average forward velocity without stumbling
  * $E_{\text{torque}}$: Energy expenditure ($\sum |\tau|^2$, penalizing spastic flailing)
  * $H_{\text{health}}$: Remaining homeostatic metabolic viability
- The top 20% of the population form the breeding elite.
- Offspring receive the adapted base weights combined with genetic mutations, seeding the next generation with hardcoded instincts.

---

## 5. Roadmap: Evolution in the Future Open Voxel World

In the upcoming voxel world sandbox:
1. **Autonomous Consensual Mating**: Agents that maintain high stamina and hunger satisfaction can choose to mate when within physical proximity of another agent.
2. **Speciation & Lineage Tracking**: Visual colors (like the neon palette in the Ghost Swarm) reflect genetic lineage clusters. Divergent species will evolve specialized adaptations (e.g., lightweight agile climbers vs. heavy high-torque sprinters).
3. **Environmental Selective Pressure**: Seasonal changes, terrain variations, and food availability will naturally weed out maladaptive genotypes without requiring artificial reward engineering.
