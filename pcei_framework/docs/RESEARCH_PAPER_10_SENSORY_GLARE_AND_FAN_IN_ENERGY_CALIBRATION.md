# Sensory Glare and Fan-In Energy Calibration: Preserving Proprioceptive Reflexes Under High-Dimensional Exteroceptive Upgrades

**Author**: Emerson Ramirez  
**Framework**: Physically Constrained Emergent Intelligence (PCEI) v0.2.2 (Embodied Platform)  
**Target Checkpoint**: `pcei_ninja_best.pt` ($572$-Dimensional Observation Space, $512$-Beam Solid-State LiDAR)  
**Classification**: Sensory Fusion, Exteroceptive Scaling, & Dynamic Range Normalization  
**Date**: October 2026  

---

## Abstract

When upgrading embodied neural agents from low-dimensional sensory proxies to high-resolution exteroceptive sensors (e.g., expanding from 16-channel 1D terrain height probes to 512-beam 3D solid-state angular LiDAR), naive linear projection layers suffer from a catastrophic failure mode we define as **Sensory Glare**. Because activation variances in fully connected layers scale proportionally with input feature dimensionality $N$, projecting an uncalibrated 512-dimensional range grid alongside a 57-dimensional proprioceptive vector produced an activation norm ratio of **$67.2\times$** ($\|\mathbf{a}_{\text{vis}}\|_2 = 932.65$ vs. $\|\mathbf{a}_{\text{prop}}\|_2 = 13.89$). This exteroceptive blowout saturated downstream non-linearities and drowned vestibular balance feedback, causing immediate motor collapse ($100\%$ failure within $2.1\text{ seconds}$, wild arm flailing, and knee buckling). In this paper, we derive the **Fan-In Energy Calibration Law**, proving that scaling exteroceptive projection weights by $S = \sqrt{N_{\text{old}} / N_{\text{new}}} \cdot \kappa$ (where $\kappa = 0.18$) combined with zero-centered range normalization ($\tilde{d} \in [-0.5, +0.5]$) restores the sensory energy equilibrium ($\|\mathbf{a}_{\text{vis}}\|_2 = 2.50$, an $18\%$ ratio). We present empirical telemetry validating that surgical warm-starting under this calibration allowed the 2.19M-parameter humanoid to seamlessly assimilate full 3D spatial vision without retraining its motor foundation, preserving continuous bipedal sprint locomotion ($v_x = 4.06\text{ m/s}$).

---

## 1. Introduction & The Sensory Glare Pathology

Embodied agents navigating complex obstacle courses require high-fidelity spatial exteroception to detect elevation changes, gaps, and approaching obstacles. In early training iterations, the Embodied PCEI humanoid utilized a simplified 16-channel 1D height-probe pseudo-vision vector:
$$\mathbf{s}_t = [\mathbf{x}_{\text{prop}} \in \mathbb{R}^{57}, \mathbf{x}_{\text{probe}} \in \mathbb{R}^{16}, \mathbf{x}_{\text{meta}} \in \mathbb{R}^3] \in \mathbb{R}^{76}$$

To navigate multi-tier obstacles (Stage 1 Quad Steps, Stage 2 Balance Beam, and Stage 4 Warped Wall), the sensory apparatus was upgraded to a head-mounted solid-state angular LiDAR casting $512$ rays ($16$ vertical elevation rows $\times$ $32$ horizontal azimuth columns):
$$\mathbf{s}_t = [\mathbf{x}_{\text{prop}} \in \mathbb{R}^{57}, \mathbf{x}_{\text{lidar}} \in \mathbb{R}^{512}, \mathbf{x}_{\text{meta}} \in \mathbb{R}^3] \in \mathbb{R}^{572}$$

```
   NAIVE EXPANSION (SENSORY GLARE)             CALIBRATED FAN-IN PROJECTION
   
   [ 57 Proprio ] ──► [ 13.89 Norm ]            [ 57 Proprio ] ──► [ 13.89 Norm ]
                             ▲                                           ▲
                             │ 67.2x DROWN                               │ 0.18x EQUILIBRIUM
                             ▼                                           ▼
   [ 512 LiDAR ]  ──► [ 932.65 Norm ]           [ 512 LiDAR ]  ──► [  2.50 Norm ]
   (Torso Collapse, Arm Flailing)               (Balanced Vestibular Reflexes)
```

### 1.1 The Empirical Breakdown
Upon expanding the observation dimension from $76$ to $572$ without energy calibration:
1. **Reflex Drowning**: The humanoid instantly collapsed upon spawning at $X=0.0\text{m}$. Average survival dropped from $>400$ timesteps to $<15$ timesteps.
2. **Motor Seizure**: Arm actuators ($15\text{--}20$) saturated at extreme boundaries ($\pm 1.0$), while knee and hip flexion torques chattered violently.
3. **Loss of Vestibular Sensitivity**: Despite receiving non-zero gravity and angular velocity telemetry ($\mathbf{x}_{\text{prop}}$), the policy outputs remained completely unresponsive to torso tilt.

---

## 2. Mathematical Formulation

### 2.1 Forward Activation Variance in Mixed-Modality Projections
Consider the first sensory encoder layer $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, where $\mathbf{W} \in \mathbb{R}^{D \times N_{\text{obs}}}$ with $D=128$. Partition the input and weight matrix into proprioceptive ($p$) and visual/exteroceptive ($v$) blocks:
$$\mathbf{z} = \mathbf{W}_p \mathbf{x}_p + \mathbf{W}_v \mathbf{x}_v + \mathbf{W}_m \mathbf{x}_m + \mathbf{b}$$
where $N_p = 57$ and $N_v = 512$.

Assuming zero-mean independent components with variances $\sigma_{x_p}^2, \sigma_{x_v}^2$ and weight initializations $\sigma_{w_p}^2, \sigma_{w_v}^2$, the expected $L_2$ norm squared of the visual activation contribution is:
$$\mathbb{E}\left[\|\mathbf{a}_v\|_2^2\right] = \mathbb{E}\left[\|\mathbf{W}_v \mathbf{x}_v\|_2^2\right] = D \cdot N_v \cdot \sigma_{w_v}^2 \cdot \sigma_{x_v}^2$$

When $N_v$ expands from $16 \to 512$ ($32\times$ increase) while retaining weight magnitudes $\sigma_{w_v} \approx \sigma_{w_p}$:
$$\frac{\mathbb{E}[\|\mathbf{a}_v\|_2^2]}{\mathbb{E}[\|\mathbf{a}_p\|_2^2]} = \frac{N_v \cdot \sigma_{w_v}^2 \cdot \sigma_{x_v}^2}{N_p \cdot \sigma_{w_p}^2 \cdot \sigma_{x_p}^2} \propto \frac{512}{57} \approx 8.98$$

### 2.2 The Positive Range DC Offset Problem
In standard raycasting, distances are normalized to unit range $d_k / d_{\max} \in [0, 1]$. Because obstacles and floors are rarely at distance zero, the expected value of open space rays is close to $1.0$:
$$\mathbb{E}[x_{v, k}] = \mu_v \approx 0.85 \gg 0$$
The expected activation vector contains a massive DC bias:
$$\mathbb{E}[\mathbf{a}_v] = \mathbf{W}_v \mathbb{E}[\mathbf{x}_v] = \mu_v \sum_{k=1}^{N_v} \mathbf{w}_{v, k}$$
For $N_v = 512$, this summation sums $512$ column vectors, producing an activation shift of norm $\|\mathbb{E}[\mathbf{a}_v]\|_2 \approx \mathcal{O}(N_v \cdot \mu_v \cdot \sigma_w)$. This drove post-layer LayerNorm and SiLU gates into complete saturation.

---

## 3. The Fan-In Energy Calibration Law

To eliminate Sensory Glare and establish physiological equilibrium between exteroceptive foresight and proprioceptive balance reflexes, we formulate two mandatory mathematical corrections:

### 3.1 Zero-Centered Range Normalization
Each raw LiDAR beam distance $d_k \in [0, d_{\max}]$ (where $d_{\max} = 10.0\text{ meters}$) is normalized via a zero-centered affine transformation:
$$\tilde{d}_k = \left( \frac{\min(d_k, d_{\max})}{d_{\max}} \right) - 0.5 \in [-0.5, +0.5]$$
Under this formulation:
* Clear unobstructed air ($d_k = 10.0\text{m}$) evaluates to $+0.5$.
* Immediate obstacle impact ($d_k = 0.0\text{m}$) evaluates to $-0.5$.
* Intermediate free field evaluates to $\mathbb{E}[\tilde{d}_k] \approx 0.0$.
This strictly guarantees $\mathbb{E}[\mathbf{a}_v] \approx \mathbf{0}$, eliminating DC bias accumulation.

### 3.2 Fan-In Energy Scaling
To preserve the dynamic range of core proprioception, the total energy of the exteroceptive subspace must be bounded relative to vestibular energy:
$$\|\mathbf{a}_v\|_2 \le \kappa \cdot \|\mathbf{a}_p\|_2$$
where empirical calibration establishes the optimal sensory coupling factor $\kappa = 0.18$.

When transferring weights from an older $N_{\text{old}}$-dimensional sensory proxy to an $N_{\text{new}}$-dimensional sensor, the projection matrix is calibrated according to:
$$\mathbf{W}_v^{\text{new}} = \left( \sqrt{\frac{N_{\text{old}}}{N_{\text{new}}}} \right) \cdot \kappa \cdot \mathbf{W}_v^{\text{centroid}} + \boldsymbol{\epsilon}$$
$$\boldsymbol{\epsilon} \sim \mathcal{N}\left(0, \left(0.005 \cdot \sqrt{\frac{N_{\text{old}}}{N_{\text{new}}}} \cdot \kappa\right)^2\right)$$

#### Theorem 1 (Vestibular Preservation Under High-Dimensional Expansion)
Let $\mathbf{W}_p$ produce nominal proprioceptive activation $\mathbf{a}_p$ with variance $\sigma_p^2$. If $\mathbf{W}_v^{\text{new}}$ is scaled according to $S = \sqrt{\frac{N_{\text{old}}}{N_{\text{new}}}} \cdot \kappa$, the signal-to-noise ratio (SNR) of vestibular balance feedback through LayerNorm satisfies:
$$\operatorname{SNR}_{\text{vestibular}} \ge \frac{\sigma_p^2}{\sigma_p^2 + \kappa^2 \sigma_p^2} = \frac{1}{1 + \kappa^2}$$
For $\kappa = 0.18$, $\operatorname{SNR}_{\text{vestibular}} \ge 0.968$, guaranteeing that vestibular balance signals retain $>96.8\%$ of their effective representational power.

---

## 4. Empirical Telemetry & Ablation Ledger

The calibration protocol was evaluated on checkpoint `pcei_ninja_best.pt`. Sensory activation norms and agent kinematics were audited before and after applying the Fan-In Calibration Law.

### Table 1: Sensory Activation Norms Before and After Calibration
| Metric / Layer Channel | Uncalibrated (Sensory Glare) | Calibrated ($\kappa = 0.18$, Zero-Centered) | Theoretical Target | Delta / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Proprioception Norm ($\|\mathbf{a}_p\|_2$)** | $13.89$ | $13.89$ | $13.89$ | Float-exact invariant ($100\%$) |
| **Exteroception Norm ($\|\mathbf{a}_v\|_2$)** | **$932.65$** | **$2.50$** | $2.50$ | **$-99.73\%$ (Glare Extinguished)** |
| **Metabolic Norm ($\|\mathbf{a}_m\|_2$)** | $0.84$ | $0.84$ | $0.84$ | Invariant |
| **Ratio ($\|\mathbf{a}_v\|_2 / \|\mathbf{a}_p\|_2$)** | **$67.2\times$ (Blown Out)**| **$0.18\times$** | $0.18\times$ | Balanced biological ratio |
| **Mean Arm Torque Magnitude** | $0.842$ (Flailing) | $0.081$ (Stabilized) | $<0.15$ | Flailing suppressed by $90.4\%$ |
| **Mean Leg Torque Magnitude** | $0.985$ (Saturated) | $0.342$ (Dynamic) | $\approx 0.35$ | Dynamic range recovered |
| **50-Step Survival Rate (Dead-Stop)** | **$0\%$ ($14$ steps)** | **$100\%$ ($50/50$)** | $100\%$ | Instant postural recovery |
| **Upright Torso Z Height** | $0.42\text{m}$ (Collapsed) | **$1.31\text{m}$ (Upright)** | $>1.25\text{m}$ | Full bipedal stance restored |

```
   Downstream Torso Height Over Time (First 50 Steps from Dead-Stop)
   
   Height (m)
   1.40 ┤        ┌──────────────────────────────── Calibrated (Z ≈ 1.31m Upright)
   1.20 ┤       ┌┘
   1.00 ┤      ┌┘
   0.80 ┤     ┌┘
   0.60 ┤    ┌┘
   0.40 ┤────┴──────┐
   0.20 ┤           └───────────────────────────── Uncalibrated (Collapsed at step 14)
   0.00 ┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──►
        0  4  8 12 16 20 24 28 32 36 40 44 48 Timesteps
```

---

## 5. Surgical Warm-Starting Protocol

Rather than re-initializing the policy network (which would destroy millions of steps of learned bipedal balance), we established the following non-destructive weight surgery protocol:

1. **Extract Proprioceptive Core**: Clone weights $\mathbf{W}_{:, 0:57}$ and biases directly from the donor checkpoint.
2. **Extract Metabolic Core**: Clone weights $\mathbf{W}_{:, -3:}$ float-exact.
3. **Centroid Ray Projection**: Compute the spatial centroid of previous 1D probes $\bar{\mathbf{w}}_{\text{old}} = \frac{1}{N_{\text{old}}} \sum_{j} \mathbf{w}_{\text{old}, j}$.
4. **Calibrated Fan-In Injection**: Tile the centroid across all $512$ 3D rays scaled by $S_{\text{fan-in}}$, perturbing with low-amplitude Gaussian noise $\sigma = 0.005 \cdot S_{\text{fan-in}}$ to break rotational symmetry.
5. **Cold-Start Critic / Warm-Start Actor**: Re-initialize the PPO optimizer state to discard stale momentum vectors while preserving the actor's policy distribution.

---

## 6. Conclusion

High-dimensional exteroceptive sensory expansion in embodied foundation models is subject to the mathematical hazard of Sensory Glare, wherein uncalibrated channel counts dominate vestibular balance loops. By establishing the **Fan-In Energy Calibration Law** ($\mathbf{W}_v \propto \sqrt{N_{\text{old}}/N_{\text{new}}} \cdot 0.18$) paired with zero-centered range normalization ($[-0.5, +0.5]$), sensory equilibrium is maintained. This protocol enables robotic platforms to upgrade from crude proximity sensors to high-density 3D LiDAR vision without inducing motor amnesia or catastrophic postural collapse.

---

## References

1. He, K., et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ICCV*, 1026–1034.
2. Ramirez, E. (2026). Cross-Modal Synergy vs. The Sensory Dependency Effect. *PCEI Research Compendium Paper III*.
3. Todorov, E., et al. (2012). MuJoCo: A physics engine for model-based control. *IEEE/RSJ IROS*, 5026–5033.
