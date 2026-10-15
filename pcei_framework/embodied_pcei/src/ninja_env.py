"""
embodied_pcei/src/ninja_env.py

High-performance physical simulation environment for the American Ninja Warrior course.
Built on MuJoCo with native Apple Silicon performance.

Features:
- Articulated bipedal humanoid with 21 joint torque actuators.
- Staged American Ninja Warrior course:
  * Stage 0: Flat sprint runway (0 to 8m)
  * Stage 1: Quad steps / angled stepping stones (8 to 18m)
  * Stage 2: Elevated balance beam (18 to 28m)
  * Stage 3: Hurdles & stepping barriers (28 to 38m)
  * Stage 4: Warped wall incline ramp & summit buzzer (38 to 48m)
- Raycast Pseudo-Vision: forward height probes measuring terrain/obstacle topology.
- Metabolic / Homeostatic Telemetry: lungs/stamina, stomach/hunger, heart/health.
- Built-in offscreen renderer for live web dashboard streaming.
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Dict, Tuple, Optional

XML_PATH = Path(__file__).resolve().parent.parent / "models" / "ninja_course.xml"
FLAT_ARENA_XML = Path(__file__).resolve().parent.parent / "models" / "flat_arena.xml"

# Obstacle Stage Milestones along +X
STAGE_MILESTONES = [
    (8.0,  "Stage 1: Quad Steps"),
    (18.0, "Stage 2: Balance Beam"),
    (28.0, "Stage 3: Hurdles"),
    (38.0, "Stage 4: Warped Wall"),
    (46.0, "Course Finish: Summit Buzzer"),
]

# 18 Progressive Navigation Waypoints across the 48m Ninja Warrior Course
# Format: (Target 3D Coordinate [X, Y, Z], Waypoint Label, Arrival Radius in meters)
COURSE_WAYPOINTS = [
    (np.array([3.0,  0.00, 1.33], dtype=np.float32), "Runway Accelerate 1",          1.2),
    (np.array([6.0,  0.00, 1.33], dtype=np.float32), "Runway Accelerate 2",          1.2),
    (np.array([8.0,  0.00, 1.33], dtype=np.float32), "Runway Exit Threshold",        1.0),
    (np.array([10.0, -0.60, 0.20], dtype=np.float32), "Stage 1: Quad Step 1 (Right)", 0.8),
    (np.array([12.2,  0.60, 0.30], dtype=np.float32), "Stage 1: Quad Step 2 (Left)",  0.8),
    (np.array([14.4, -0.50, 0.35], dtype=np.float32), "Stage 1: Quad Step 3 (Right)", 0.8),
    (np.array([16.6,  0.50, 0.40], dtype=np.float32), "Stage 1: Quad Step 4 (Left)",  0.8),
    (np.array([18.5,  0.00, 0.45], dtype=np.float32), "Platform Mid 1",               1.0),
    (np.array([20.5,  0.00, 0.50], dtype=np.float32), "Stage 2: Balance Beam Entry",  0.6),
    (np.array([23.5,  0.00, 0.50], dtype=np.float32), "Stage 2: Balance Beam Mid",    0.6),
    (np.array([26.5,  0.00, 0.50], dtype=np.float32), "Stage 2: Balance Beam Exit",   0.6),
    (np.array([28.5,  0.00, 0.50], dtype=np.float32), "Platform Mid 2",               1.0),
    (np.array([31.5,  0.00, 0.20], dtype=np.float32), "Stage 3: Hurdle 1",            0.9),
    (np.array([34.5,  0.00, 0.28], dtype=np.float32), "Stage 3: Hurdle 2",            0.9),
    (np.array([37.5,  0.00, 0.22], dtype=np.float32), "Stage 3: Hurdle 3",            0.9),
    (np.array([40.0,  0.00, 0.30], dtype=np.float32), "Stage 4: Warped Wall Approach",1.2),
    (np.array([42.5,  0.00, 0.70], dtype=np.float32), "Stage 4: Warped Wall Incline", 1.2),
    (np.array([47.5,  0.00, 2.45], dtype=np.float32), "Course Finish: Summit Buzzer", 0.6),
]

# Translucent Neon Color Palette for Multi-Agent Ghost Swarm Rendering
GHOST_PALETTE = [
    np.array([0.2, 0.7, 1.0, 0.16], dtype=np.float32),  # Neon Cyan
    np.array([0.7, 0.3, 1.0, 0.16], dtype=np.float32),  # Electric Purple
    np.array([0.1, 0.9, 0.6, 0.16], dtype=np.float32),  # Neon Emerald
    np.array([1.0, 0.55, 0.1, 0.16], dtype=np.float32), # Sunset Orange
    np.array([1.0, 0.2, 0.6, 0.16], dtype=np.float32),  # Neon Pink
    np.array([0.35, 0.8, 1.0, 0.16], dtype=np.float32), # Sky Blue
    np.array([0.85, 0.95, 0.2, 0.16], dtype=np.float32),# Lime Volt
]

# Canonical Reference Quiet Standing Equilibrium Pose (21 Actuated Joints)
# Torso at Z=1.168m, spine vertical, knees naturally flexed at 0.05 rad (~3 deg), arms down at sides
STAND_REF_QPOS = np.array([
    0.0, 0.0, 0.0,                      # abdomen_z, abdomen_y, abdomen_x
    0.0, 0.0, -0.05, 0.05, -0.025, 0.0, # right_hip_x, z, y, knee, ankle_y, ankle_x
    0.0, 0.0, -0.05, 0.05, -0.025, 0.0, # left_hip_x, z, y, knee, ankle_y, ankle_x
    0.65, 0.0, 0.0,                     # right_shoulder_x, y, elbow (down at side)
    -0.65, 0.0, 0.0,                    # left_shoulder_x, y, elbow (down at side)
], dtype=np.float32)

STAND_JOINT_WEIGHTS = np.array([
    1.5, 1.5, 1.5,                      # abdomen_z, abdomen_y, abdomen_x
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,       # right leg (hip_x, hip_z, hip_y, knee, ankle_y, ankle_x)
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,       # left leg (hip_x, hip_z, hip_y, knee, ankle_y, ankle_x)
    0.5, 0.5, 0.5,                      # right arm (shoulder_x, shoulder_y, elbow)
    0.5, 0.5, 0.5                       # left arm (shoulder_x, shoulder_y, elbow)
], dtype=np.float32)


class NinjaEnv:
    def __init__(self, xml_path: Optional[str] = None, frame_skip: int = 5, arena_mode: bool = True, stream_mode: str = "cruise"):
        self.arena_mode = arena_mode
        self.stream_mode = stream_mode  # "cruise", "stand", "getup", "reach"
        if xml_path is None:
            xml_path = str(FLAT_ARENA_XML if arena_mode else XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        
        self.dt = self.model.opt.timestep * frame_skip # 0.005 * 5 = 0.025s (40 Hz)
        
        self.n_actuators = self.model.nu # 21
        self.n_qpos = self.model.nq     # 28 (7 root + 21 joints)
        self.n_qvel = self.model.nv     # 27 (6 root + 21 joints)
        
        # Solid-State Angular LiDAR Sensor Configuration
        # 16 vertical beams (pitch: -35 deg to +15 deg) x 32 horizontal beams (yaw: -45 deg to +45 deg) = 512 rays
        self.num_lidar_v = 16
        self.num_lidar_h = 32
        self.num_lidar_rays = self.num_lidar_v * self.num_lidar_h # 512
        self.num_rays = self.num_lidar_rays # backwards-compatibility alias
        self.lidar_max_range = 10.0 # meters
        self.lidar_v_angles = np.radians(np.linspace(-35, 15, self.num_lidar_v))
        self.lidar_h_angles = np.radians(np.linspace(-45, 45, self.num_lidar_h))
        
        # Precompute unit direction vectors in head local coordinate frame
        # Forward = +X, Left = +Y, Up = +Z
        local_vectors = []
        for pitch in self.lidar_v_angles:
            for yaw in self.lidar_h_angles:
                vx = np.cos(pitch) * np.cos(yaw)
                vy = np.cos(pitch) * np.sin(yaw)
                vz = np.sin(pitch)
                v = np.array([vx, vy, vz], dtype=np.float64)
                v /= np.linalg.norm(v)
                local_vectors.append(v)
        self.lidar_local_vectors = np.array(local_vectors, dtype=np.float64) # (128, 3)
        self.current_lidar_grid = np.zeros((self.num_lidar_v, self.num_lidar_h), dtype=np.float32)
        self._geomid_buf = np.zeros(1, dtype=np.int32)
        self._visor_offset = np.array([0.15, 0.0, 0.0], dtype=np.float64)
        
        # Metabolic internal states
        self.stamina = 1.0 # 0.0 to 1.0
        self.health = 1.0  # 0.0 to 1.0
        self.hunger = 0.0  # 0.0 to 1.0
        
        # Milestone tracking & Course Waypoints
        self.milestones_reached = set()
        self.waypoints_reached = set()
        self.current_wp_idx = 0
        self.furthest_x = 0.0
        self.max_steps = 1000

        # Flat Arena & Dynamic Random Goal Navigation State (Stream 4: Cruise)
        self.goal_pos = np.array([5.0, 0.0, 1.25], dtype=np.float32)
        self.goals_completed = 0
        self.goal_arrival_radius = 2.0      # Wide capture radius (2.0m): easily clears by walking near target
        self.min_dist_to_goal = 5.0
        self.prev_dist_to_goal = 5.0
        self.steps_not_approaching = 0
        self.transition_leeway_window = 90  # ~2.25s leeway (90 steps at 40Hz) for turning & realignment
        self.goal_transition_steps = 0      # No leeway on initial spawn (must advance immediately)

        # Stream 1: Quiet Standing Stillness State
        self.stand_still_steps = 0
        self.stand_drift_steps = 0
        self.stand_one_leg_steps = 0

        # Stream 2: Fall Recovery State (Ragdoll & Curated Library)
        self.fall_state_type = "random_ragdoll"
        self.consecutive_upright_steps = 0
        self.prev_z = 0.20
        self.prev_head_z = 0.20
        self.getup_stagnant_steps = 0
        self.max_elevation_achieved = 0.20
        self.max_head_elevation = 0.20
        self.getup_goal_pos = np.array([0.0, 0.0, 1.168], dtype=np.float32)
        self.getup_success_count = 0

        # Stream 3: Bilateral Hand Reaching State (Bolted Torso)
        self.active_reach_hand = "right"
        self.reach_target_pos = np.array([0.35, -0.25, 1.35], dtype=np.float32)
        self.reach_touches = 0
        self.prev_reach_dist = 0.40
        self.min_reach_dist = 0.40
        self.steps_on_reach_target = 0
        self.steps_receding_reach = 0
        
        # Bilateral gait stance duty-cycle counters
        self.r_foot_contact_steps = 0
        self.l_foot_contact_steps = 0
        
        # Offscreen renderer instance (lazy-loaded)
        self._renderer = None
        
        # Cache body and geom IDs for fast lookups
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.head_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.r_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.l_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.r_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        self.l_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
        self.r_hand_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_hand")
        self.l_hand_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand")
        self.r_arm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_lower_arm")
        self.l_arm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_lower_arm")

        # Cache leg joint qpos indices for athletic stance and gait monitoring
        self.r_knee_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")]
        self.l_knee_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")]
        self.r_hip_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_y")]
        self.l_hip_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_y")]
        self.r_ankle_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_y")]
        self.l_ankle_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_y")]
        # Cache abdomen joint qpos indices for core posture alignment
        self.abdomen_x_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "abdomen_x")]
        self.abdomen_y_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "abdomen_y")]
        self.abdomen_z_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "abdomen_z")]

        # Cache arm joint qpos and dof velocity indices for bilateral reach
        self.arm_qpos_indices = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_x")],
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_y")],
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow")],
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_x")],
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")],
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow")],
        ]
        self.arm_dof_indices = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_x")],
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_y")],
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow")],
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_x")],
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")],
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow")],
        ]

        # Cache collision geoms for physical impact detection (head, visor, chest, torso)
        # All geoms belonging to world body (body 0) are environmental surfaces: floor + all obstacles
        self.world_obstacle_geoms = {i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == 0}
        self.body_impact_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "chest_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "visor"),
        }

        # Cache humanoid geom indices (all geoms belonging to non-world bodies) for ghost rendering
        self.humanoid_geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] != 0]

        # Cache limb geoms for Get-Up multi-phase floor contact righting
        self.r_shin_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_shin_geom")
        self.l_shin_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_shin_geom")
        self.support_hand_geoms = {
            self.r_hand_geom_id,
            self.l_hand_geom_id,
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_larm_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_larm_geom"),
        }
        self.support_knee_geoms = {self.r_shin_geom_id, self.l_shin_geom_id}

    def _lock_torso_for_reach(self):
        """Bolts root torso, core spine, and legs in neutral athletic stance (Stream 3: Isolated Arm Motor Cortex)."""
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 1.35
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Upright quaternion
        self.data.qvel[0:6] = 0.0  # Zero base linear & angular velocity
        # Lock core spine neutral
        self.data.qpos[7:10] = 0.0  # abdomen_z, abdomen_y, abdomen_x
        # Lock legs in neutral supportive stance
        self.data.qpos[10:22] = 0.0 # zero all hip, knee, ankle
        self.data.qpos[self.r_knee_qpos] = 0.30
        self.data.qpos[self.l_knee_qpos] = 0.30
        self.data.qpos[self.r_ankle_qpos] = -0.10
        self.data.qpos[self.l_ankle_qpos] = -0.10
        self.data.qvel[6:21] = 0.0  # Zero core & lower body velocities (DOFs 6 to 20, preserving arm DOFs 21:27)

    def _sample_reach_target(self):
        """
        Samples a randomized 3D reach goal via Forward Kinematics (FK) from the active arm's
        reachable joint space, mathematically guaranteeing 100.0% of targets are physically reachable (Stream 3).
        """
        self.active_reach_hand = np.random.choice(["right", "left"])
        
        # Save current posture
        saved_qpos = self.data.qpos.copy()
        target_pos = None

        # Sample joint angles from the arm's valid forward/lateral reachable envelope
        for _ in range(25):
            if self.active_reach_hand == "right":
                # Right arm envelope: shoulder roll/pitch and elbow flexion
                target_sx = float(np.random.uniform(-0.75, 0.40))
                target_sy = float(np.random.uniform(-0.60, 0.85))
                target_el = float(np.random.uniform(-1.10, 0.45))
                self.data.qpos[self.arm_qpos_indices[0]] = target_sx  # right_shoulder_x
                self.data.qpos[self.arm_qpos_indices[1]] = target_sy  # right_shoulder_y
                self.data.qpos[self.arm_qpos_indices[2]] = target_el  # right_elbow
                mujoco.mj_forward(self.model, self.data)
                cand_pos = self.data.geom_xpos[self.r_hand_geom_id].copy()
            else:
                # Left arm envelope: shoulder roll/pitch and elbow flexion
                target_sx = float(np.random.uniform(-0.40, 0.75))
                target_sy = float(np.random.uniform(-0.60, 0.85))
                target_el = float(np.random.uniform(-0.45, 1.10))
                self.data.qpos[self.arm_qpos_indices[3]] = target_sx  # left_shoulder_x
                self.data.qpos[self.arm_qpos_indices[4]] = target_sy  # left_shoulder_y
                self.data.qpos[self.arm_qpos_indices[5]] = target_el  # left_elbow
                mujoco.mj_forward(self.model, self.data)
                cand_pos = self.data.geom_xpos[self.l_hand_geom_id].copy()

            # Restore original pose to measure distance
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)
            curr_h = self.data.geom_xpos[self.r_hand_geom_id] if self.active_reach_hand == "right" else self.data.geom_xpos[self.l_hand_geom_id]
            dist_to_cand = float(np.linalg.norm(curr_h - cand_pos))
            target_pos = cand_pos
            # Ensure target is at least 0.16m away so the agent must actively articulate its arm
            if dist_to_cand >= 0.16:
                break

        # Always restore state cleanly
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)

        self.reach_target_pos = target_pos.astype(np.float32)
        self.goal_pos = self.reach_target_pos.copy()

    def _apply_fall_state(self, pose_name: Optional[str] = None):
        """
        Initializes agent in a randomized fall state with 360° randomized yaw,
        perturbed joints, physics contact settling, and a randomized standing destination goal (Stream 2).
        """
        if pose_name is None:
            poses = ["prone", "supine", "lateral_left", "lateral_right", "kneeling_crumpled", "seated_slumped", "random_ragdoll"]
            pose_name = np.random.choice(poses)
        self.fall_state_type = pose_name

        self.data.qvel[:] = 0.0
        # Random initial floor translation offset
        spawn_x = float(np.random.uniform(-0.4, 0.4))
        spawn_y = float(np.random.uniform(-0.4, 0.4))

        # Base orientation quaternion [w, x, y, z] per canonical pose
        if pose_name == "prone":
            # Lying on stomach: pitch 90 deg forward
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, 0.22])
            q_base = np.array([0.7071, 0.0, 0.7071, 0.0])
            self.data.qpos[self.r_knee_qpos] = 0.2
            self.data.qpos[self.l_knee_qpos] = 0.2
        elif pose_name == "supine":
            # Lying on back: pitch 90 deg backward
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, 0.22])
            q_base = np.array([0.7071, 0.0, -0.7071, 0.0])
            self.data.qpos[self.r_knee_qpos] = 0.3
            self.data.qpos[self.l_knee_qpos] = 0.3
        elif pose_name == "lateral_left":
            # Lying on left flank: roll 90 deg left
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, 0.22])
            q_base = np.array([0.7071, -0.7071, 0.0, 0.0])
        elif pose_name == "lateral_right":
            # Lying on right flank: roll 90 deg right
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, 0.22])
            q_base = np.array([0.7071, 0.7071, 0.0, 0.0])
        elif pose_name == "kneeling_crumpled":
            # Crumpled on knees, torso pitched forward
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, 0.48])
            q_base = np.array([0.9238, 0.0, 0.3826, 0.0])
            self.data.qpos[self.r_knee_qpos] = 1.7
            self.data.qpos[self.l_knee_qpos] = 1.7
            self.data.qpos[self.r_hip_qpos] = -1.0
            self.data.qpos[self.l_hip_qpos] = -1.0
        elif pose_name == "seated_slumped":
            # Slumped seated on buttocks, legs forward
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, 0.35])
            q_base = np.array([0.9238, 0.0, -0.3826, 0.0])
            self.data.qpos[self.r_hip_qpos] = -1.4
            self.data.qpos[self.l_hip_qpos] = -1.4
            self.data.qpos[self.r_knee_qpos] = 0.2
            self.data.qpos[self.l_knee_qpos] = 0.2
        else:
            # Random ragdoll drop: drop from 0.45m with completely randomized 3D orientation
            self.data.qpos[0:3] = np.array([spawn_x, spawn_y, float(np.random.uniform(0.40, 0.60))])
            rand_quat = np.random.uniform(-1.0, 1.0, size=4)
            q_base = rand_quat / np.linalg.norm(rand_quat)

        # Apply randomized 360° yaw rotation to base pose quaternion
        yaw = float(np.random.uniform(-np.pi, np.pi))
        q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
        # Quaternion multiplication: q_yaw * q_base
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_base
        q_rot = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)
        self.data.qpos[3:7] = q_rot / np.linalg.norm(q_rot)

        # Stochastic joint perturbation across all 21 joint degrees of freedom
        self.data.qpos[7:] += np.random.uniform(-0.20, 0.20, size=self.n_actuators)

        # Settle contact forces smoothly (12 passive sub-steps with zero actuation)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(12):
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Sample a randomized standing destination goal beacon (0.6m - 1.8m away at random heading)
        stand_dist = float(np.random.uniform(0.6, 1.8))
        stand_heading = float(np.random.uniform(-np.pi, np.pi))
        cur_torso = self.data.xpos[self.torso_body_id]
        gx = float(cur_torso[0] + stand_dist * np.cos(stand_heading))
        gy = float(cur_torso[1] + stand_dist * np.sin(stand_heading))
        gz = 1.168  # Grounded standing torso height equilibrium
        self.getup_goal_pos = np.array([gx, gy, gz], dtype=np.float32)
        self.goal_pos = self.getup_goal_pos.copy()

    def _sample_new_goal(self, is_initial: bool = False):
        """Samples a new random navigation target within the 50m x 50m flat arena."""
        torso_pos = self.data.xpos[self.torso_body_id]
        if is_initial:
            # Initial target: distance 5.0m - 7.5m in forward cone (-60 deg to +60 deg)
            dist = float(np.random.uniform(5.0, 7.5))
            angle = float(np.random.uniform(-np.pi / 3, np.pi / 3))
            self.goal_transition_steps = 0  # Spawn directly facing target: immediate approach required
        else:
            # Subsequent target: distance 5.5m - 9.0m in full 360 deg omnidirectional bearing
            dist = float(np.random.uniform(5.5, 9.0))
            angle = float(np.random.uniform(-np.pi, np.pi))
            self.goal_transition_steps = self.transition_leeway_window  # ~1.85s direction change leeway
            
        gx = float(np.clip(torso_pos[0] + dist * np.cos(angle), -22.0, 22.0))
        gy = float(np.clip(torso_pos[1] + dist * np.sin(angle), -22.0, 22.0))

        # Ensure newly sampled target is at least (arrival_radius + 1.5m = 3.5m) away from current torso
        for _ in range(20):
            if np.linalg.norm(np.array([gx - torso_pos[0], gy - torso_pos[1]])) >= (self.goal_arrival_radius + 1.5):
                break
            center_angle = float(np.arctan2(-torso_pos[1], -torso_pos[0]))
            angle = float(center_angle + np.random.uniform(-np.pi / 4, np.pi / 4))
            dist = float(np.random.uniform(5.5, 8.5))
            gx = float(np.clip(torso_pos[0] + dist * np.cos(angle), -22.0, 22.0))
            gy = float(np.clip(torso_pos[1] + dist * np.sin(angle), -22.0, 22.0))

        gz = 1.25  # Torso/chest level
        self.goal_pos = np.array([gx, gy, gz], dtype=np.float32)
        self.min_dist_to_goal = float(np.linalg.norm(self.goal_pos - torso_pos))
        self.prev_dist_to_goal = float(self.min_dist_to_goal)
        self.steps_since_progress = 0
        self.steps_not_approaching = 0
        self.negative_vx_steps = 0

    @property
    def observation_dim(self) -> int:
        """
        Total observation vector length:
        - Torso height z: 1
        - Torso orientation (forward & up vectors): 6
        - Torso linear velocity (local/global): 3
        - Torso angular velocity: 3
        - 21 joint angles: 21
        - 21 joint velocities: 21
        - 2 foot contact forces: 2
        - 512 solid-state LiDAR range rays (16x32 grid): 512
        - 3 ego-centric relative waypoint vector (Δx, Δy, Δz local): 3
        - 3 metabolic stats (stamina, health, hunger): 3
        Total: 1 + 6 + 3 + 3 + 21 + 21 + 2 + 512 + 3 + 3 = 575
        """
        return 575

    @property
    def action_dim(self) -> int:
        return self.n_actuators # 21

    def reset(self, seed: Optional[int] = None, spawn_x: float = 0.0, initial_vx: float = 0.0) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial root pose: at X=spawn_x, Y=0, Z=1.33 (athletic bent-knee stance)
        self.data.qpos[0] = spawn_x   # X
        self.data.qpos[1] = 0.0       # Y
        self.data.qpos[2] = 1.33      # Z
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0]) # Quaternion (unit)
        
        # Staggered lead leg initialization: alternates 50% left lead, 50% right lead
        # This breaks single-leg asymmetric gait traps and forces both thighs to cycle
        lead_leg = np.random.choice(["right", "left"])
        if lead_leg == "right":
            self.data.qpos[self.r_hip_qpos] = -0.35  # Right thigh poised forward
            self.data.qpos[self.r_knee_qpos] = 0.45
            self.data.qpos[self.r_ankle_qpos] = -0.15
            self.data.qpos[self.l_hip_qpos] = 0.05   # Left leg supporting in stance
            self.data.qpos[self.l_knee_qpos] = 0.25
            self.data.qpos[self.l_ankle_qpos] = -0.05
        else:
            self.data.qpos[self.l_hip_qpos] = -0.35  # Left thigh poised forward
            self.data.qpos[self.l_knee_qpos] = 0.45
            self.data.qpos[self.l_ankle_qpos] = -0.15
            self.data.qpos[self.r_hip_qpos] = 0.05   # Right leg supporting in stance
            self.data.qpos[self.r_knee_qpos] = 0.25
            self.data.qpos[self.r_ankle_qpos] = -0.05
        
        # Add slight stochastic noise to joint angles for exploration diversity
        joint_noise = np.random.uniform(-0.03, 0.03, size=self.n_actuators)
        self.data.qpos[7:] += joint_noise
        
        # Zero velocities with tiny noise (or inject forward running momentum if pioneer spawn)
        self.data.qvel[:] = np.random.uniform(-0.02, 0.02, size=self.n_qvel)
        if initial_vx > 0.0:
            self.data.qvel[0] = initial_vx + np.random.uniform(-0.1, 0.1)
        
        # Reset internal stats
        self.stamina = 1.0
        self.health = 1.0
        self.hunger = 0.0
        
        self.milestones_reached = set()
        self.waypoints_reached = set()
        self.current_wp_idx = 0

        if self.stream_mode == "stand":
            # Stream 1: Quiet Standing Balance (Canonical Reference Equilibrium Stance)
            self.data.qpos[0] = 0.0
            self.data.qpos[1] = 0.0
            self.data.qpos[2] = 1.168
            self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            self.data.qvel[:] = 0.0
            self.data.qpos[7:7 + self.n_actuators] = STAND_REF_QPOS.copy()
            # Add slight stochastic exploration noise (+/- 0.015 rad ~ 0.8 deg) so agent learns to balance
            self.data.qpos[7:7 + self.n_actuators] += np.random.uniform(-0.015, 0.015, size=self.n_actuators)
            # Settle foot ground contacts smoothly (10 passive sub-steps with zero actuation)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(10):
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.goal_pos = np.array([0.0, 0.0, 1.168], dtype=np.float32)
            self.stand_still_steps = 0
            self.stand_drift_steps = 0
            self.stand_one_leg_steps = 0
            self.goals_completed = 0
        elif self.stream_mode == "getup":
            # Stream 2: Fall Recovery (Randomized 360° poses & standing destination goal)
            self._apply_fall_state()
            self.prev_z = float(self.data.xpos[self.torso_body_id][2])
            self.prev_head_z = float(self.data.xpos[self.head_body_id][2])
            self.consecutive_upright_steps = 0
            self.getup_stagnant_steps = 0
            self.max_elevation_achieved = self.prev_z
            self.max_head_elevation = self.prev_head_z
            self.getup_success_count = 0
            self.goals_completed = 0
        elif self.stream_mode == "reach":
            # Stream 2: Bilateral Hand Reaching (Bolted Torso with neutral resting arm configuration)
            self._lock_torso_for_reach()
            # Initialize both arms down at sides (natural resting posture at hip)
            self.data.qpos[self.arm_qpos_indices[0]] = 0.65 + float(np.random.uniform(-0.03, 0.03))  # right_shoulder_x (down at side)
            self.data.qpos[self.arm_qpos_indices[1]] = float(np.random.uniform(-0.03, 0.03))         # right_shoulder_y
            self.data.qpos[self.arm_qpos_indices[2]] = float(np.random.uniform(-0.03, 0.03))         # right_elbow
            self.data.qpos[self.arm_qpos_indices[3]] = -0.65 + float(np.random.uniform(-0.03, 0.03)) # left_shoulder_x (down at side)
            self.data.qpos[self.arm_qpos_indices[4]] = float(np.random.uniform(-0.03, 0.03))         # left_shoulder_y
            self.data.qpos[self.arm_qpos_indices[5]] = float(np.random.uniform(-0.03, 0.03))         # left_elbow
            self.data.qvel[self.arm_dof_indices] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._sample_reach_target()
            is_right_target = (self.active_reach_hand == "right")
            active_h = self.data.geom_xpos[self.r_hand_geom_id] if is_right_target else self.data.geom_xpos[self.l_hand_geom_id]
            self.prev_reach_dist = float(np.linalg.norm(active_h - self.reach_target_pos))
            self.min_reach_dist = self.prev_reach_dist
            self.reach_touches = 0
            self.steps_on_reach_target = 0
            self.steps_receding_reach = 0
            self.goals_completed = 0
        elif self.arena_mode:
            self.goals_completed = 0
            self._sample_new_goal(is_initial=True)
        else:
            # Fast-forward waypoint index if spawning downfield
            for idx, (wp_pos, _, _) in enumerate(COURSE_WAYPOINTS):
                if wp_pos[0] > spawn_x + 0.3:
                    self.current_wp_idx = idx
                    break
                
        self.spawn_x = float(spawn_x)
        self.furthest_x = float(spawn_x)
        self.current_step = 0
        self.steps_since_progress = 0
        self.last_progress_x = float(spawn_x)
        self.negative_vx_steps = 0
        self.steps_not_approaching = 0
        self.r_foot_contact_steps = 0
        self.l_foot_contact_steps = 0
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        # 1. Torso state
        torso_xpos = self.data.xpos[self.torso_body_id] # (3,)
        torso_xmat = self.data.xmat[self.torso_body_id].reshape(3, 3) # (3, 3)
        forward_vec = torso_xmat[:, 0] # Local X axis in world frame
        up_vec = torso_xmat[:, 2]      # Local Z axis in world frame
        
        torso_cvel = self.data.cvel[self.torso_body_id] # 6D: [ang_vel(3), lin_vel(3)]
        ang_vel = torso_cvel[:3]
        lin_vel = torso_cvel[3:]
        
        # 2. Joint states (21 angles, 21 velocities)
        joint_qpos = self.data.qpos[7:7 + self.n_actuators]
        joint_qvel = self.data.qvel[6:6 + self.n_actuators]
        
        # 3. Touch sensors & Task-Specific Contact Indicators
        touch_forces = np.zeros(2, dtype=np.float32)
        if self.stream_mode == "reach":
            # For isolated Reach, legs are bolted stationary. Repurpose touch_forces as explicit Active Hand Indicator:
            # [1.0, 0.0] -> Right Hand Active | [0.0, 1.0] -> Left Hand Active
            if self.active_reach_hand == "right":
                touch_forces[0] = 1.0
                touch_forces[1] = 0.0
            else:
                touch_forces[0] = 0.0
                touch_forces[1] = 1.0
        else:
            if self.model.nsensor >= 2:
                touch_forces[0] = np.clip(self.data.sensordata[0] / 100.0, 0.0, 5.0) # Right foot
                touch_forces[1] = np.clip(self.data.sensordata[1] / 100.0, 0.0, 5.0) # Left foot
            
        # 4. True Solid-State Angular LiDAR (512 rays from visor/eyes)
        lidar_obs = self._compute_lidar_vision()
        
        # 5. Dynamic Goal / Progressive Waypoint (Ego-centric local relative vector)
        if self.stream_mode == "stand":
            rel_world = self.goal_pos - torso_xpos
            rel_local = torso_xmat.T @ rel_world
            waypoint_obs = np.array([
                np.clip(rel_local[0] / 10.0, -1.0, 1.0),
                np.clip(rel_local[1] / 5.0, -1.0, 1.0),
                np.clip(rel_local[2] / 2.0, -1.0, 1.0),
            ], dtype=np.float32)
        elif self.stream_mode == "getup":
            rel_world = self.getup_goal_pos - torso_xpos
            rel_local = torso_xmat.T @ rel_world
            waypoint_obs = np.array([
                np.clip(rel_local[0] / 5.0, -1.0, 1.0),
                np.clip(rel_local[1] / 5.0, -1.0, 1.0),
                np.clip(rel_local[2] / 2.0, -1.0, 1.0),
            ], dtype=np.float32)
        elif self.stream_mode == "reach":
            # Direct closed-loop motor servoing: vector from ACTIVE HAND directly to the reach target orb!
            active_h = self.data.geom_xpos[self.r_hand_geom_id] if (self.active_reach_hand == "right") else self.data.geom_xpos[self.l_hand_geom_id]
            rel_hand_world = self.reach_target_pos - active_h
            rel_hand_local = torso_xmat.T @ rel_hand_world
            waypoint_obs = np.array([
                np.clip(rel_hand_local[0] / 0.6, -1.0, 1.0),
                np.clip(rel_hand_local[1] / 0.6, -1.0, 1.0),
                np.clip(rel_hand_local[2] / 0.6, -1.0, 1.0),
            ], dtype=np.float32)
        elif self.arena_mode:
            target_pos = self.goal_pos
            rel_world = target_pos - torso_xpos
            rel_local = torso_xmat.T @ rel_world
            waypoint_obs = np.array([
                np.clip(rel_local[0] / 10.0, -1.0, 1.0),
                np.clip(rel_local[1] / 5.0, -1.0, 1.0),
                np.clip(rel_local[2] / 2.0, -1.0, 1.0),
            ], dtype=np.float32)
        else:
            target_pos = COURSE_WAYPOINTS[self.current_wp_idx][0]
            rel_world = target_pos - torso_xpos
            rel_local = torso_xmat.T @ rel_world
            waypoint_obs = np.array([
                np.clip(rel_local[0] / 10.0, -1.0, 1.0),
                np.clip(rel_local[1] / 5.0, -1.0, 1.0),
                np.clip(rel_local[2] / 2.0, -1.0, 1.0),
            ], dtype=np.float32)

        # 6. Metabolic state (stamina, health, hunger/task-channel)
        if self.stream_mode == "reach":
            hand_indicator = 1.0 if (self.active_reach_hand == "right") else -1.0
            metabolic_obs = np.array([self.stamina, self.health, hand_indicator], dtype=np.float32)
        else:
            metabolic_obs = np.array([self.stamina, self.health, self.hunger], dtype=np.float32)
        
        obs = np.concatenate([
            np.array([torso_xpos[2]], dtype=np.float32), # Torso Z (1)
            forward_vec.astype(np.float32),              # 3
            up_vec.astype(np.float32),                   # 3
            lin_vel.astype(np.float32),                  # 3
            ang_vel.astype(np.float32),                  # 3
            joint_qpos.astype(np.float32),               # 21
            joint_qvel.astype(np.float32),               # 21
            touch_forces,                                # 2
            lidar_obs,                                   # 512
            waypoint_obs,                                # 3
            metabolic_obs,                               # 3
        ])
        
        return np.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)

    def _compute_lidar_vision(self) -> np.ndarray:
        """
        True Solid-State Angular LiDAR scan from the robot's visor/eyes.
        Casts 128 rays (8 vertical elevation x 16 horizontal azimuth) using MuJoCo's native C raycaster.
        Returns normalized range vector (128,) in [0.0, 1.0].
        """
        head_pos = self.data.xpos[self.head_body_id]
        head_mat = self.data.xmat[self.head_body_id].reshape(3, 3)
        
        # Ray origin: 0.15m forward from head center through visor
        ray_origin = np.ascontiguousarray(head_pos + head_mat @ self._visor_offset, dtype=np.float64)
        
        # Transform all 128 precomputed local vectors to world coordinates
        # (128, 3) @ (3, 3).T -> (128, 3)
        world_vectors = self.lidar_local_vectors @ head_mat.T
        
        normalized_ranges = np.zeros(self.num_lidar_rays, dtype=np.float32)
        raw_grid = np.zeros((self.num_lidar_v, self.num_lidar_h), dtype=np.float32)
        
        ray_idx = 0
        for row in range(self.num_lidar_v):
            for col in range(self.num_lidar_h):
                vec = np.ascontiguousarray(world_vectors[ray_idx], dtype=np.float64)
                dist = mujoco.mj_ray(self.model, self.data, ray_origin, vec, None, 1, -1, self._geomid_buf)
                
                if dist >= 0:
                    gid = int(self._geomid_buf[0])
                    if gid in self.humanoid_geoms:
                        actual_dist = self.lidar_max_range
                    else:
                        actual_dist = min(dist, self.lidar_max_range)
                else:
                    actual_dist = self.lidar_max_range
                    
                raw_grid[row, col] = actual_dist
                # Zero-centered range: -0.5 (impact / 0m) to +0.5 (clear / 10m), eliminates DC bias
                normalized_ranges[ray_idx] = (actual_dist / self.lidar_max_range) - 0.5
                ray_idx += 1
                
        self.current_lidar_grid = raw_grid
        return normalized_ranges

    def _compute_pseudo_vision(self, agent_x: float, agent_y: float, agent_z: float) -> np.ndarray:
        """Backwards compatibility fallback for 1D pseudo-vision."""
        return self._compute_lidar_vision()[:16]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        # Clip action to [-1.0, 1.0]
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        if self.stream_mode == "reach":
            # For isolated arm reaching, freeze motor commands to core and legs
            self.data.ctrl[0:15] = 0.0

        # Substep physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.stream_mode == "reach":
                self._lock_torso_for_reach()
            
        # Extract state measurements
        torso_xpos = self.data.xpos[self.torso_body_id]
        x, y, z = torso_xpos[0], torso_xpos[1], torso_xpos[2]
        
        torso_cvel = self.data.cvel[self.torso_body_id]
        vx = torso_cvel[3] # Forward linear velocity
        vy = torso_cvel[4]
        
        torso_xmat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        up_z = torso_xmat[2, 2] # Cosine of torso vertical tilt
        
        # Update metabolic stats
        torque_cost = np.sum(np.square(action))
        self.stamina = np.clip(self.stamina - (torque_cost * 0.0003) + 0.0005, 0.0, 1.0)
        self.hunger = np.clip(self.hunger + 0.0001, 0.0, 1.0)
        
        # Multi-limb ground contact detection for floor righting and locomotion
        contact_r = False
        contact_l = False
        contact_hand = False
        contact_knee = False
        for c_idx in range(self.data.ncon):
            c = self.data.contact[c_idx]
            g1, g2 = c.geom1, c.geom2
            if (g1 == self.r_foot_geom_id and g2 in self.world_obstacle_geoms) or (g2 == self.r_foot_geom_id and g1 in self.world_obstacle_geoms):
                contact_r = True
            if (g1 == self.l_foot_geom_id and g2 in self.world_obstacle_geoms) or (g2 == self.l_foot_geom_id and g1 in self.world_obstacle_geoms):
                contact_l = True
            if (g1 in self.support_hand_geoms and g2 in self.world_obstacle_geoms) or (g2 in self.support_hand_geoms and g1 in self.world_obstacle_geoms):
                contact_hand = True
            if (g1 in self.support_knee_geoms and g2 in self.world_obstacle_geoms) or (g2 in self.support_knee_geoms and g1 in self.world_obstacle_geoms):
                contact_knee = True

        touch_r = contact_r or (bool(self.data.sensordata[0] > 5.0) if self.model.nsensor >= 1 else False)
        touch_l = contact_l or (bool(self.data.sensordata[1] > 5.0) if self.model.nsensor >= 2 else False)
        if touch_r:
            self.r_foot_contact_steps += 1
        if touch_l:
            self.l_foot_contact_steps += 1

        upright_scale = float(np.clip((z - 0.45) / 0.40, 0.0, 1.0)) # 0 at z<=0.45m, 1.0 at z>=0.85m
        in_leeway = False
        torso_speed = float(np.linalg.norm(torso_cvel[3:]))

        # ── STREAM 1: QUIET STANDING STILLNESS (REFERENCE POSE SIMILARITY) ────────
        if self.stream_mode == "stand":
            head_z = float(self.data.xpos[self.head_body_id][2])
            dist_from_origin = float(np.linalg.norm([x, y]))
            
            # 1. Joint Configuration Similarity:
            # Gaussian kernel comparing 21 actuated joints against canonical STAND_REF_QPOS
            current_qpos = self.data.qpos[7:7 + self.n_actuators]
            diff_q = current_qpos - STAND_REF_QPOS
            weighted_q_mse = float(np.sum(STAND_JOINT_WEIGHTS * (diff_q ** 2)) / np.sum(STAND_JOINT_WEIGHTS))
            r_pose = float(np.exp(-3.0 * weighted_q_mse))

            # 2. Torso Center-of-Mass Height Similarity (centered at z* = 1.168m):
            r_height = float(np.exp(-30.0 * ((z - 1.168) ** 2)))

            # 3. Torso Upright Orientation Similarity:
            r_upright = float(np.exp(-12.0 * (max(0.0, 1.0 - up_z) ** 2)))

            # 4. Multi-Rigid-Body Stillness Similarity (root linear, root angular, joint vel):
            lin_vel_sq = float(np.sum(torso_cvel[3:] ** 2))
            ang_vel_sq = float(np.sum(torso_cvel[:3] ** 2))
            joint_vel_sq = float(np.mean(self.data.qvel[6:6 + self.n_actuators] ** 2))
            r_stillness = float(np.exp(-4.0 * (lin_vel_sq + 0.5 * ang_vel_sq + 0.1 * joint_vel_sq)))

            # 5. Foot Ground Contact Multiplier:
            both_feet_down = bool(touch_r and touch_l)
            one_foot_down = bool(touch_r or touch_l)
            if both_feet_down:
                m_feet = 1.0
            elif one_foot_down:
                m_feet = 0.45
            else:
                m_feet = 0.05

            # 6. Action Control / Effort Penalty:
            r_ctrl = -0.002 * float(np.sum(action ** 2))

            # Bounded Gaussian Similarity Reward (Peak = 14.0/step at quiet standing equilibrium)
            total_reward = m_feet * (4.0 * r_pose + 3.0 * r_height + 3.0 * r_upright + 4.0 * r_stillness) + r_ctrl

            # Track stillness success criteria:
            is_stand_still = bool(
                r_pose > 0.85 and
                r_height > 0.88 and
                r_upright > 0.90 and
                r_stillness > 0.80 and
                both_feet_down
            )
            if is_stand_still:
                self.stand_still_steps += 1
            else:
                self.stand_still_steps = max(0, self.stand_still_steps - 1)

            # Check collision with environmental obstacles/ground using body geoms (head, chest, torso, visor)
            is_obstacle_or_ground_impact = False
            for c_idx in range(self.data.ncon):
                c = self.data.contact[c_idx]
                if (c.geom1 in self.body_impact_geoms and c.geom2 in self.world_obstacle_geoms) or \
                   (c.geom2 in self.body_impact_geoms and c.geom1 in self.world_obstacle_geoms):
                    is_obstacle_or_ground_impact = True
                    break

            terminated = False
            # Clean early termination only when genuinely fallen or severely displaced:
            if z < 0.85 or up_z < 0.55 or is_obstacle_or_ground_impact:
                terminated = True
                total_reward -= 15.0
            elif dist_from_origin > 1.20:
                terminated = True
                total_reward -= 10.0

            # Mastery time limit: 384 steps (~9.6s)
            truncated = (self.current_step >= 384)
            if truncated and (not terminated) and z > 1.10 and both_feet_down:
                total_reward += 50.0  # Mastery bonus for surviving 384 steps upright on both feet

            # Detailed metrics for logging & dashboard
            torso_roll = abs(float(torso_xmat[2, 1]))
            torso_pitch = abs(float(torso_xmat[2, 0]))
            spine_lateral = abs(float(self.data.qpos[self.abdomen_x_qpos]))
            spine_sagittal = abs(float(self.data.qpos[self.abdomen_y_qpos]))

            info = {
                "x_pos": float(x), "y_pos": float(y), "z_pos": float(z),
                "vx": float(vx), "torso_speed": float(torso_speed),
                "furthest_x": float(self.furthest_x), "milestones": 0,
                "waypoint_idx": 0, "waypoint_name": "Quiet Stand Stillness",
                "dist_to_wp": float(dist_from_origin), "waypoints_completed": int(self.stand_still_steps > 150),
                "goals_completed": int(self.stand_still_steps > 150),
                "dist_to_goal": float(dist_from_origin),
                "goal_arrival_radius": 1.20,
                "goal_pos": [0.0, 0.0, 1.168],
                "target_local_pos": [0.0, 0.0, 0.0],
                "gaze_alignment": 1.0,
                "stamina": float(self.stamina), "health": float(self.health), "hunger": float(self.hunger),
                "step": self.current_step, "lidar_grid": self.current_lidar_grid.tolist(),
                "in_leeway": False, "leeway_remaining": 0,
                "stream_mode": "stand", "active_reach_hand": "none", "reach_touches": 0,
                "getup_success": False,
                "both_feet_down": bool(both_feet_down),
                "one_foot_down": bool(one_foot_down),
                "stand_still_score": float(is_stand_still),
                "torso_roll": float(torso_roll),
                "torso_pitch": float(torso_pitch),
                "spine_lateral": float(spine_lateral),
                "spine_sagittal": float(spine_sagittal),
                "r_pose": float(r_pose),
                "r_height": float(r_height),
                "r_upright": float(r_upright),
                "r_stillness": float(r_stillness),
            }
            return self._get_obs(), float(total_reward), terminated, truncated, info

        # ── STREAM 2: FALL RECOVERY (GET-UP) ────────────────────────────────
        elif self.stream_mode == "getup":
            head_z = float(self.data.xpos[self.head_body_id][2])
            delta_head_z = head_z - self.prev_head_z
            self.prev_head_z = head_z

            r_foot_z = float(self.data.xpos[self.r_foot_id][2])
            l_foot_z = float(self.data.xpos[self.l_foot_id][2])
            max_foot_z = max(r_foot_z, l_foot_z)

            # 1. REWARD FOR HEAD & TORSO ELEVATION (Rises vertically, head leads above torso):
            reward_head_elevation = 16.0 * delta_head_z
            reward_torso_elevation = 12.0 * max(0.0, z - self.prev_z)
            reward_head_height = 5.0 * max(0.0, head_z - 0.40)
            reward_head_above_torso = 3.5 * max(0.0, head_z - z)
            reward_upright = 3.0 * max(0.0, up_z - 0.30)

            # 2. MULTI-PHASE GROUND CONTACT SUPPORT (Hands, Knees, Feet):
            both_feet_down = touch_r and touch_l
            one_foot_down = touch_r or touch_l

            # Floor bracing support (head_z < 1.15m): reward hands pushing floor & knees tucked
            reward_hand_brace = 4.0 if (contact_hand and head_z < 1.15) else 0.0
            reward_knee_brace = 3.0 if (contact_knee and head_z < 1.15) else 0.0

            # Standing stage (head_z >= 1.05m): reward planting feet
            if both_feet_down:
                reward_feet_ground = 6.0
            elif one_foot_down:
                reward_feet_ground = 2.0
            elif head_z >= 1.20:
                reward_feet_ground = -1.0 # Only penalize missing foot contact once upright
            else:
                reward_feet_ground = 0.0  # Zero penalty while floor-bound rolling or pushing up!

            # Floor proximity for feet during ground phase
            reward_feet_down = 2.0 * max(0.0, 0.30 - max_foot_z) if head_z < 1.15 else 0.0

            # 3. MITIGATE LEG AIR-FLAILING:
            # Only penalize feet higher than head IF head is actually on the ground and feet are violently kicked up
            feet_above_head = max(0.0, max_foot_z - max(0.40, head_z))
            penalty_shooting_legs = -4.0 * feet_above_head

            # Air-flail penalty only when feet are high in air with zero contact
            penalty_air_flail = -2.0 * max(0.0, max_foot_z - 0.50) if (head_z < 1.0 and not (contact_hand or contact_knee)) else 0.0

            # 4. Energy cost & Arm Windmilling Suppression
            arm_flail_penalty = -0.001 * float(np.sum(np.square(action[15:21])))
            reward_ctrl = -0.001 * torque_cost + arm_flail_penalty
            dist_to_stand_goal = float(np.linalg.norm(self.getup_goal_pos[:2] - np.array([x, y])))
            reward_goal_heading = 2.0 * max(0.0, 1.0 - dist_to_stand_goal / 2.0) if head_z > 1.30 else 0.0

            total_reward = (
                reward_head_elevation +
                reward_torso_elevation +
                reward_head_height +
                reward_head_above_torso +
                reward_upright +
                reward_hand_brace +
                reward_knee_brace +
                reward_feet_ground +
                reward_feet_down +
                penalty_shooting_legs +
                penalty_air_flail +
                reward_ctrl +
                reward_goal_heading
            )
            self.prev_z = z

            # Track peak head elevation for stagnation monitoring
            if head_z > self.max_head_elevation:
                self.max_head_elevation = head_z
                self.getup_stagnant_steps = 0
            else:
                self.getup_stagnant_steps += 1

            # Upright standing with feet planted verification:
            # Requires head up (> 1.35m), torso upright (z > 1.05m, up_z > 0.70), and foot contact
            is_upright = (head_z > 1.35) and (z > 1.05) and (up_z > 0.70) and (one_foot_down or (head_z > 1.45 and z > 1.10))
            if is_upright:
                self.consecutive_upright_steps += 1
                total_reward += 3.0
                if both_feet_down:
                    total_reward += 3.0  # Extra jackpot bonus when BOTH feet are planted!
            else:
                self.consecutive_upright_steps = 0

            # Lenient Ground Collapse Stagnation & Rise Stall Deaths:
            # Tolerates 140 steps (~3.5s) struggling on ground without head elevation
            # Tolerates 180 steps (~4.5s) coordinating knee-tuck and feet planting to reach upright stance
            is_ground_stuck = (head_z < 0.50 and self.getup_stagnant_steps > 140)
            is_stall_stuck = (head_z >= 0.50 and self.consecutive_upright_steps == 0 and self.getup_stagnant_steps > 180)

            terminated = False
            if self.consecutive_upright_steps >= 12:
                total_reward += 100.0  # Upright Recovery Jackpot!
                self.getup_success_count += 1
                terminated = True
            elif is_ground_stuck or is_stall_stuck:
                terminated = True
                total_reward -= 10.0  # Timeout stagnation death

            truncated = (self.current_step >= 280)
            target_pos = self.getup_goal_pos

            info = {
                "x_pos": float(x), "y_pos": float(y), "z_pos": float(z),
                "vx": float(vx), "torso_speed": float(torso_speed),
                "furthest_x": float(self.furthest_x), "milestones": 0,
                "waypoint_idx": 0, "waypoint_name": f"Get-Up ({self.fall_state_type})",
                "dist_to_wp": float(max(0.0, 1.55 - head_z)),
                "waypoints_completed": int(self.consecutive_upright_steps >= 12),
                "goals_completed": int(self.consecutive_upright_steps >= 12),
                "dist_to_goal": float(dist_to_stand_goal),
                "goal_arrival_radius": 0.50,
                "goal_pos": self.getup_goal_pos.tolist(),
                "target_local_pos": (torso_xmat.T @ (target_pos - torso_xpos)).tolist(),
                "gaze_alignment": float(max(0.0, up_z)),
                "stamina": float(self.stamina), "health": float(self.health), "hunger": float(self.hunger),
                "step": self.current_step, "lidar_grid": self.current_lidar_grid.tolist(),
                "in_leeway": False, "leeway_remaining": 0,
                "stream_mode": "getup", "active_reach_hand": "none", "reach_touches": 0,
                "getup_success": bool(self.consecutive_upright_steps >= 12),
                "head_z": float(head_z),
                "both_feet_down": bool(both_feet_down),
            }
            return self._get_obs(), float(total_reward), terminated, truncated, info

        # ── STREAM 2: BILATERAL HAND REACHING (BOLTED BODY) ────────────────
        elif self.stream_mode == "reach":
            # Torso, core, and legs are bolted in neutral athletic upright stance.
            # Control is isolated strictly to the 6 arm actuators reaching for 3D orbs.
            r_h = self.data.geom_xpos[self.r_hand_geom_id]
            l_h = self.data.geom_xpos[self.l_hand_geom_id]
            dist_r = float(np.linalg.norm(r_h - self.reach_target_pos))
            dist_l = float(np.linalg.norm(l_h - self.reach_target_pos))

            # The designated active hand is the hand closer to the target ball
            is_right_target = (self.active_reach_hand == "right")

            if is_right_target:
                active_h = r_h
                wrong_h = l_h
                dist_to_target = dist_r
                dist_inactive = dist_l
                active_body_id = self.r_arm_body_id
                wrong_body_id = self.l_arm_body_id
                active_act = action[15:18]
                wrong_act = action[18:21]
                active_qvel = self.data.qvel[self.arm_dof_indices[0:3]]
                wrong_qvel = self.data.qvel[self.arm_dof_indices[3:6]]
                wrong_qpos = self.data.qpos[self.arm_qpos_indices[3:6]]
                wrong_rest_qpos = np.array([-0.65, 0.0, 0.0], dtype=np.float32)
            else:
                active_h = l_h
                wrong_h = r_h
                dist_to_target = dist_l
                dist_inactive = dist_r
                active_body_id = self.l_arm_body_id
                wrong_body_id = self.r_arm_body_id
                active_act = action[18:21]
                wrong_act = action[15:18]
                active_qvel = self.data.qvel[self.arm_dof_indices[3:6]]
                wrong_qvel = self.data.qvel[self.arm_dof_indices[0:3]]
                wrong_qpos = self.data.qpos[self.arm_qpos_indices[0:3]]
                wrong_rest_qpos = np.array([0.65, 0.0, 0.0], dtype=np.float32)

            # 1. Main reach distance progress and proximity reward
            dist_prog = self.prev_reach_dist - dist_to_target
            reward_reach_prog = 25.0 * dist_prog
            reward_proximity = 6.0 * np.exp(-4.0 * dist_to_target)

            # 2. MASSIVE SPEED EFFICIENCY REWARDS:
            # a) Direct closing velocity vector along the unit ray connecting active hand to target orb
            delta_active_target = self.reach_target_pos - active_h
            dist_hand_target = max(1e-4, float(np.linalg.norm(delta_active_target)))
            u_reach_target = delta_active_target / dist_hand_target
            active_hand_vel = self.data.cvel[active_body_id][3:6]
            v_closing = float(np.dot(active_hand_vel, u_reach_target))
            # Speed efficiency bonus for thrusting active hand toward target
            reward_closing_speed = 4.0 * max(0.0, v_closing)

            # b) Mild latency penalty (encourages prompt reaching)
            penalty_latency = -0.02 * (float(self.steps_on_reach_target) / 40.0)

            # 3. Slight reward for moving the correct hand
            active_hand_speed = float(np.linalg.norm(active_hand_vel))
            active_qvel_mag = float(np.sum(np.abs(active_qvel)))
            reward_correct_motion = 0.15 * min(1.0, active_hand_speed / 0.25) + 0.10 * np.tanh(active_qvel_mag)

            # 4. REGULARIZATION ON INACTIVE HAND (stay neutral and relaxed at side)
            wrong_hand_speed = float(np.linalg.norm(self.data.cvel[wrong_body_id][3:6]))
            penalty_wrong_act = -0.05 * float(np.sum(np.square(wrong_act)))
            penalty_wrong_qvel = -0.05 * float(np.sum(np.abs(wrong_qvel)))
            penalty_wrong_speed = -0.20 * min(1.0, wrong_hand_speed)
            penalty_wrong_raised = -0.50 * max(0.0, float(wrong_h[2]) - 1.28)
            penalty_wrong_posture = -0.20 * min(1.0, float(np.linalg.norm(wrong_qpos - wrong_rest_qpos)))
            penalty_wrong_lead = -0.50 * max(0.0, dist_to_target - dist_inactive)
            penalty_wrong_encroach = -0.50 * max(0.0, 0.35 - dist_inactive)
            penalty_wrong_touch = -2.0 if dist_inactive < 0.14 else 0.0

            # General control penalty on active arm to encourage efficient articulation
            reward_ctrl = -0.001 * float(np.sum(np.square(active_act)))

            self.steps_on_reach_target += 1
            if dist_to_target < self.min_reach_dist:
                self.min_reach_dist = dist_to_target
                self.steps_receding_reach = 0
            elif dist_to_target > self.min_reach_dist + 0.25:
                self.steps_receding_reach += 1

            reward_touch = 0.0
            speed_efficiency_jackpot = 0.0
            if dist_to_target < 0.14:
                # 1. Base touch award
                base_touch = 40.0
                # 2. Speed efficiency jackpot: maximum when acquired in few steps
                speed_efficiency_jackpot = max(0.0, 110.0 - float(self.steps_on_reach_target)) * 0.40
                # 3. Consecutive hit streak multiplier
                touch_streak_bonus = min(30.0, float(self.reach_touches) * 6.0)
                reward_touch = base_touch + speed_efficiency_jackpot + touch_streak_bonus

                self.reach_touches += 1
                self._sample_reach_target()
                is_right_target = (self.active_reach_hand == "right")
                active_h = self.data.geom_xpos[self.r_hand_geom_id] if is_right_target else self.data.geom_xpos[self.l_hand_geom_id]
                dist_to_target = float(np.linalg.norm(active_h - self.reach_target_pos))
                # Reset timers for next target!
                self.steps_on_reach_target = 0
                self.steps_receding_reach = 0
                self.min_reach_dist = dist_to_target

            self.prev_reach_dist = dist_to_target
            total_reward = (
                reward_reach_prog +
                reward_proximity +
                reward_closing_speed +
                reward_correct_motion +
                reward_touch +
                penalty_latency +
                penalty_wrong_act +
                penalty_wrong_qvel +
                penalty_wrong_speed +
                penalty_wrong_raised +
                penalty_wrong_posture +
                penalty_wrong_lead +
                penalty_wrong_encroach +
                penalty_wrong_touch +
                reward_ctrl
            )

            # Lenient Reaching Time Limits:
            # Tolerates 110 steps (~2.75s) per target orb, or 45 steps receding away
            is_reach_timeout = (self.steps_on_reach_target > 110)
            is_reach_receding = (self.steps_receding_reach > 45)

            terminated = False
            if is_reach_timeout or is_reach_receding:
                terminated = True
                total_reward -= 2.0  # Mild timeout penalty

            truncated = (self.current_step >= 384)
            target_pos = self.reach_target_pos

            head_xmat = self.data.xmat[self.head_body_id].reshape(3, 3)
            head_fwd = head_xmat[:, 0]
            delta_head_target = target_pos - self.data.xpos[self.head_body_id]
            cos_gaze = float(np.dot(head_fwd, delta_head_target / max(1e-3, np.linalg.norm(delta_head_target))))

            info = {
                "x_pos": float(x), "y_pos": float(y), "z_pos": float(z),
                "vx": float(vx), "torso_speed": float(torso_speed),
                "furthest_x": float(self.furthest_x), "milestones": 0,
                "waypoint_idx": 0, "waypoint_name": f"Reach ({self.active_reach_hand.upper()})",
                "dist_to_wp": float(dist_to_target), "waypoints_completed": int(self.reach_touches),
                "goals_completed": int(self.reach_touches),
                "dist_to_goal": float(dist_to_target),
                "goal_arrival_radius": 0.14,
                "goal_pos": self.reach_target_pos.tolist(),
                "target_local_pos": (torso_xmat.T @ (target_pos - torso_xpos)).tolist(),
                "gaze_alignment": float(cos_gaze),
                "stamina": float(self.stamina), "health": float(self.health), "hunger": float(self.hunger),
                "step": self.current_step, "lidar_grid": self.current_lidar_grid.tolist(),
                "in_leeway": False, "leeway_remaining": 0,
                "stream_mode": "reach", "active_reach_hand": str(self.active_reach_hand),
                "reach_touches": int(self.reach_touches), "getup_success": False, "stand_still_score": 0.0,
                "wrong_hand_dist": float(dist_inactive), "wrong_hand_touch": bool(dist_inactive < 0.14),
                "wrong_hand_speed": float(wrong_hand_speed),
                "wrong_arm_action_penalty": float(penalty_wrong_act),
                "wrong_arm_raised_penalty": float(penalty_wrong_raised),
                "v_closing": float(v_closing),
                "reward_closing_speed": float(reward_closing_speed),
                "speed_efficiency_jackpot": float(speed_efficiency_jackpot),
            }
            return self._get_obs(), float(total_reward), terminated, truncated, info
        # Target Navigation Vector & Forward Propulsion
        if self.arena_mode:
            target_pos = self.goal_pos
            delta_target = target_pos - torso_xpos
            dist_to_target = float(np.linalg.norm(delta_target))
            dist_xy = float(np.linalg.norm(delta_target[:2]))

            # Check dynamic goal arrival (entering wide arrival radius)
            reward_arrival = 0.0
            if dist_xy < self.goal_arrival_radius:
                self.goals_completed += 1
                # Escalating jackpot for chained consecutive goals: 50 for 1st, 85 for 2nd, 120 for 3rd, 155 for 4th...
                reward_arrival = 50.0 + 35.0 * (self.goals_completed - 1)
                self._sample_new_goal(is_initial=False)
                target_pos = self.goal_pos
                delta_target = target_pos - torso_xpos
                dist_to_target = float(np.linalg.norm(delta_target))
                dist_xy = float(np.linalg.norm(delta_target[:2]))
                self.prev_dist_to_goal = dist_to_target
                dist_delta = 0.0
            else:
                dist_delta = self.prev_dist_to_goal - dist_to_target
                self.prev_dist_to_goal = dist_to_target

            # Leeway period active after goal completion to allow decelerating & turning
            in_leeway = (self.goal_transition_steps > 0)
            if in_leeway:
                self.goal_transition_steps -= 1
                # While turning during leeway, continuously update closest distance baseline to current position
                self.min_dist_to_goal = dist_to_target

            # 1. Distance progress potential (scaled with goal streak) & sharp penalty for moving away
            streak_mult = 1.0 + 0.35 * min(4, self.goals_completed)
            if dist_delta >= 0.0:
                reward_dist_progress = 3.0 * streak_mult * dist_delta
            else:
                # Double penalty for moving further away from target outside of turn leeway
                reward_dist_progress = 0.0 if in_leeway else (6.0 * dist_delta)

            # 2. Penalty for drifting/retreating beyond the closest distance achieved so far (muted in leeway)
            retreat_distance = max(0.0, dist_to_target - self.min_dist_to_goal)
            penalty_retreat = 0.0 if in_leeway else (-3.0 * retreat_distance if retreat_distance > 0.10 else 0.0)

            unit_target_dir = delta_target / max(0.1, dist_to_target)
            vel_towards_target = float(np.dot(torso_cvel[3:], unit_target_dir))
            torso_speed = float(np.linalg.norm(torso_cvel[3:]))

            # 3. Controlled Forward Movement vs Severe Retrograde Penalty
            # Cap forward reward at 1.8 m/s so Olympic sprinting gives no extra reward
            v_eff = min(max(0.0, vel_towards_target), 1.8)
            reward_forward = 3.0 * v_eff * upright_scale
            penalty_retrograde = 0.0 if (vel_towards_target >= 0.0 or in_leeway) else (5.0 * vel_towards_target)

            # 4. Heading guidance velocity (steep negative slope when heading backwards outside leeway)
            if vel_towards_target >= 0.0:
                reward_heading = 1.5 * v_eff * upright_scale
            else:
                reward_heading = 0.0 if in_leeway else (3.0 * vel_towards_target)

            # 5. Speed Regulation Penalty: Penalize reckless overspeeding (> 2.0 m/s) to promote stable locomotion
            penalty_overspeed = 0.0
            if torso_speed > 2.0:
                penalty_overspeed = -3.0 * ((torso_speed - 2.0) ** 2)

            # 6. Turn alignment & braking reward during transition leeway: encourage slowing down & rapid yaw alignment to new goal
            reward_turn = 0.0
            reward_brake = 0.0
            if in_leeway:
                torso_fwd = torso_xmat[:, 0]
                cos_torso_goal = float(np.dot(torso_fwd[:2], unit_target_dir[:2]))
                # Reward rotating body to face new target (+0.0 to +1.0)
                reward_turn = 1.0 * max(0.0, cos_torso_goal)
                # Reward decelerating below 1.6 m/s so the agent can pivot cleanly
                if torso_speed < 1.6:
                    reward_brake = 0.6 * (1.6 - torso_speed)

            # 7. Subsequent Goal Urgency: Time-decay pressure once Goal 1 is cleared
            penalty_urgency = 0.0
            if self.goals_completed >= 1 and not in_leeway:
                penalty_urgency = -0.08 * min(3, self.goals_completed)

            reward_wp = (
                reward_arrival +
                reward_dist_progress +
                penalty_retreat +
                penalty_retrograde +
                penalty_overspeed +
                reward_turn +
                reward_brake +
                penalty_urgency
            )
            reward_drift = 0.0
            reward_milestone = 0.0
            dist_to_wp = dist_to_target
        else:
            # Obstacle course mode
            wp_pos, wp_name, wp_radius = COURSE_WAYPOINTS[self.current_wp_idx]
            target_pos = wp_pos
            delta_to_wp = wp_pos - torso_xpos
            dist_to_wp = float(np.linalg.norm(delta_to_wp))
            dist_xy = float(np.linalg.norm(delta_to_wp[:2]))

            reward_forward = 3.5 * vx * upright_scale
            if vx < -0.05:
                reward_forward += 1.5 * vx

            wp_reached = False
            if dist_xy < wp_radius:
                wp_reached = True
            elif torso_xpos[0] >= wp_pos[0] and abs(torso_xpos[1] - wp_pos[1]) < (wp_radius + 0.3):
                wp_reached = True
                
            reward_wp = 0.0
            if wp_reached and self.current_wp_idx not in self.waypoints_reached:
                self.waypoints_reached.add(self.current_wp_idx)
                if self.current_wp_idx == len(COURSE_WAYPOINTS) - 1:
                    reward_wp = 100.0
                elif 3 <= self.current_wp_idx <= 14:
                    reward_wp = 25.0
                else:
                    reward_wp = 15.0
                if self.current_wp_idx < len(COURSE_WAYPOINTS) - 1:
                    self.current_wp_idx += 1
                    wp_pos, wp_name, wp_radius = COURSE_WAYPOINTS[self.current_wp_idx]
                    target_pos = wp_pos
                    delta_to_wp = wp_pos - torso_xpos
                    dist_to_wp = float(np.linalg.norm(delta_to_wp))

            wp_unit_dir = delta_to_wp / max(0.1, dist_to_wp)
            vel_towards_wp = float(np.dot(torso_cvel[3:], wp_unit_dir))
            reward_heading = 2.0 * max(-0.5, min(vel_towards_wp, 3.5)) * upright_scale

            target_y = float(wp_pos[1])
            reward_drift = -0.5 * abs(y - target_y)

            reward_milestone = 0.0
            for threshold, name in STAGE_MILESTONES:
                if x >= threshold and threshold not in self.milestones_reached:
                    self.milestones_reached.add(threshold)
                    reward_milestone += 50.0

        # Head Gaze Alignment Reward: mild reward for facing objective with head/visor
        head_xmat = self.data.xmat[self.head_body_id].reshape(3, 3)
        head_fwd = head_xmat[:, 0]  # Local X axis of head (forward gaze direction)
        head_pos = self.data.xpos[self.head_body_id]
        delta_head_target = target_pos - head_pos
        dist_head_target = float(np.linalg.norm(delta_head_target))
        u_head_target = delta_head_target / max(1e-3, dist_head_target)
        cos_gaze = float(np.dot(head_fwd, u_head_target))
        reward_gaze = 0.15 * max(0.0, cos_gaze)

        # 2. Upright posture reward (staying balanced above 0.70m)
        reward_upright = 1.0 if z > 0.70 else 0.0
        
        # 3. Upright orientation bonus & active lateral roll stabilization
        roll_y = torso_xmat[1, 2]
        reward_roll = -2.0 * abs(roll_y)
        reward_orient = 0.8 * max(0.0, up_z) + reward_roll

        # Advance speed for biomechanical stepping gating
        forward_vec = torso_xmat[:, 0]
        v_advance = float(np.dot(torso_cvel[3:6], forward_vec)) if self.arena_mode else vx

        # 4. Alternating Bipedal Stepping Cadence
        reward_gait = 0.0
        if z > 0.55 and v_advance > 0.10:
            if touch_r ^ touch_l:
                reward_gait = 2.0 * min(v_advance, 1.5)
            elif touch_r and touch_l:
                reward_gait = -0.4 * min(v_advance, 1.5)

        # 5. Knee Articulation & Stance Stability
        r_knee_q = self.data.qpos[self.r_knee_qpos]
        l_knee_q = self.data.qpos[self.l_knee_qpos]
        reward_knee = 0.0
        if z > 0.55 and v_advance > 0.10:
            if not touch_r:
                if 0.25 <= r_knee_q <= 1.1:
                    reward_knee += 0.3
                elif r_knee_q > 1.6:
                    reward_knee -= 0.5 * (r_knee_q - 1.6)
            if not touch_l:
                if 0.25 <= l_knee_q <= 1.1:
                    reward_knee += 0.3
                elif l_knee_q > 1.6:
                    reward_knee -= 0.5 * (l_knee_q - 1.6)
            if touch_r and r_knee_q < 0.50:
                reward_knee += 0.2
            if touch_l and l_knee_q < 0.50:
                reward_knee += 0.2

        # 6. Swing foot forward velocity bonus
        r_foot_vfwd = float(np.dot(self.data.cvel[self.r_foot_id][3:6], forward_vec)) if self.arena_mode else self.data.cvel[self.r_foot_id][3]
        l_foot_vfwd = float(np.dot(self.data.cvel[self.l_foot_id][3:6], forward_vec)) if self.arena_mode else self.data.cvel[self.l_foot_id][3]
        reward_swing = 0.0
        if z > 0.55 and v_advance > 0.10:
            if not touch_r and r_foot_vfwd > 0.15:
                reward_swing += 0.3 * min(r_foot_vfwd, 2.0)
            if not touch_l and l_foot_vfwd > 0.15:
                reward_swing += 0.3 * min(l_foot_vfwd, 2.0)

        # 7. Thigh Forward Stride Separation
        r_hip_q = self.data.qpos[self.r_hip_qpos]
        l_hip_q = self.data.qpos[self.l_hip_qpos]
        reward_stride = 0.0
        if z > 0.55 and v_advance > 0.10:
            if not touch_r:
                stride_diff = l_hip_q - r_hip_q
                if 0.1 < stride_diff < 1.3:
                    reward_stride = 1.0 * stride_diff
            elif not touch_l:
                stride_diff = r_hip_q - l_hip_q
                if 0.1 < stride_diff < 1.3:
                    reward_stride = 1.0 * stride_diff

        # 8. Bilateral Stance Symmetry (Duty-Cycle Equalizer)
        reward_symmetry = 0.0
        total_contacts = self.r_foot_contact_steps + self.l_foot_contact_steps
        if self.current_step > 10 and total_contacts > 6 and z > 0.55:
            asym = abs(self.r_foot_contact_steps - self.l_foot_contact_steps) / total_contacts
            if asym > 0.20:
                reward_symmetry = -2.0 * (asym - 0.20)

        # 9. Arm Windmilling Suppression (Penalize flailing arms to force balance from core and legs)
        arm_flail_penalty = -0.008 * float(np.sum(np.square(action[15:21])))
        
        # 10. Energy expenditure penalty
        reward_ctrl = -0.0015 * torque_cost + arm_flail_penalty

        total_reward = (
            reward_forward +
            reward_heading +
            reward_wp +
            reward_gaze +
            reward_upright +
            reward_orient +
            reward_gait +
            reward_knee +
            reward_swing +
            reward_stride +
            reward_symmetry +
            reward_ctrl +
            reward_drift +
            reward_milestone
        )

        if self.arena_mode:
            rad_dist = float(np.linalg.norm([x, y]))
            if rad_dist > self.furthest_x:
                self.furthest_x = rad_dist

            if in_leeway:
                # During turning leeway window: reset & do not accumulate moving-away or stagnation counters
                self.steps_not_approaching = 0
                self.negative_vx_steps = 0
                self.steps_since_progress = 0
                self.min_dist_to_goal = dist_to_target
            else:
                # Active approach check: requires advancing toward target (> 0.15 m/s and closing distance)
                is_actively_approaching = (vel_towards_target >= 0.15) and (dist_delta > 0.001)
                if is_actively_approaching:
                    self.steps_not_approaching = 0
                else:
                    self.steps_not_approaching += 1

                # Receding tracker: moving away from target
                is_receding = (vel_towards_target < -0.05) or (dist_delta < -0.003)
                if is_receding:
                    self.negative_vx_steps += 1
                else:
                    self.negative_vx_steps = max(0, self.negative_vx_steps - 1)

                # Record best approach distance
                if dist_to_target < self.min_dist_to_goal:
                    self.min_dist_to_goal = dist_to_target
                    self.steps_since_progress = 0
                else:
                    self.steps_since_progress += 1
        else:
            if x > self.furthest_x:
                self.furthest_x = x

            if x > self.last_progress_x + 0.06:
                self.last_progress_x = x
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1

            if vx < -0.15:
                self.negative_vx_steps += 1
            else:
                self.negative_vx_steps = 0

        # ── TERMINATION & TRUNCATION ──────────────────────────────────────────
        # Exact preserved death conditions:
        # 1. Physical collision: head, visor, chest, or upper torso directly impacts the floor/obstacles
        # 2. True collapse: torso center of mass drops below 0.35m or head drops below 0.45m
        # 3. Complete inversion: torso tilted > 81 deg from vertical (up_z < 0.15)
        # 4. Arena boundary departure (|x| > 24m or |y| > 24m) or course lateral departure (|y| > 3.0m)
        # 5. Fast elimination when moving away or not actively approaching target (outside leeway)
        terminated = False

        is_obstacle_or_ground_impact = False
        for c_idx in range(self.data.ncon):
            c = self.data.contact[c_idx]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.body_impact_geoms and g2 in self.world_obstacle_geoms) or \
               (g2 in self.body_impact_geoms and g1 in self.world_obstacle_geoms):
                is_obstacle_or_ground_impact = True
                break

        head_z = self.data.xpos[self.head_body_id][2]
        is_altitude_collapse = (z < 0.35) or (head_z < 0.45)
        is_tilt_collapse = (up_z < 0.15)
        is_physical_fall = is_obstacle_or_ground_impact or is_altitude_collapse or is_tilt_collapse

        if self.arena_mode:
            is_boundary_departure = (abs(x) > 24.0 or abs(y) > 24.0)

            # FAST KILL TRIGGERS (active outside turning leeway period):
            # 1. Backing away kill: moved away from target for > 25 steps (~0.6s)
            is_moving_away_kill = (not in_leeway) and (self.negative_vx_steps > 25)

            # 2. Retreat distance kill: backed up > 0.40m further than closest approach achieved
            is_retreat_kill = (not in_leeway) and (dist_to_target > self.min_dist_to_goal + 0.40) and (self.current_step > 15)

            # 3. Not actively approaching kill: dawdling, circling, or stationary for > 80 steps (~2.0s)
            is_not_approaching_kill = (not in_leeway) and (self.current_step > 25 and self.steps_not_approaching > 80)

            # 4. Stagnation kill: no new record distance achieved for > 85 steps (~2.125s)
            is_stagnant_kill = (not in_leeway) and (self.current_step > 25 and self.steps_since_progress > 85)

            if is_physical_fall:
                terminated = True
                total_reward -= 20.0
                self.health = max(0.0, self.health - 0.2)
            elif is_boundary_departure:
                terminated = True
                total_reward -= 20.0
            elif is_moving_away_kill or is_retreat_kill:
                terminated = True
                total_reward -= 25.0  # Severe penalty for retreating/moving away from target
            elif is_not_approaching_kill or is_stagnant_kill:
                terminated = True
                total_reward -= 25.0  # Severe penalty for failing to actively approach target

            # Dropped Goal Streak Penalty:
            # If the agent achieved at least 1 goal, but died before reaching the next goal,
            # apply a moderate penalty to encourage multi-goal chaining without negating arrival returns
            if terminated and self.goals_completed >= 1:
                dropped_goal_penalty = -15.0 * (1.0 + 0.2 * min(3, self.goals_completed - 1))
                total_reward += dropped_goal_penalty
        else:
            is_moving_backwards = (x < self.spawn_x - 0.50) or (self.negative_vx_steps > 60)
            is_stagnant = (self.current_step > 50 and self.steps_since_progress > 90)
            is_lateral_departure = (abs(y) > 3.0)

            if is_physical_fall:
                terminated = True
                total_reward -= 20.0
                self.health = max(0.0, self.health - 0.2)
            elif is_moving_backwards:
                terminated = True
                total_reward -= 15.0
            elif is_stagnant:
                terminated = True
                total_reward -= 10.0
            elif is_lateral_departure:
                terminated = True
                total_reward -= 20.0

            if x >= 47.0:
                terminated = True
                total_reward += 200.0

        truncated = (self.current_step >= self.max_steps)

        info = {
            "x_pos": float(x),
            "y_pos": float(y),
            "z_pos": float(z),
            "vx": float(vx),
            "torso_speed": float(torso_speed),
            "furthest_x": float(self.furthest_x),
            "milestones": len(self.milestones_reached),
            "waypoint_idx": int(self.current_wp_idx),
            "waypoint_name": str(COURSE_WAYPOINTS[self.current_wp_idx][1]) if not self.arena_mode else f"Arena Target #{self.goals_completed + 1}",
            "dist_to_wp": float(dist_to_wp),
            "waypoints_completed": len(self.waypoints_reached) if not self.arena_mode else int(self.goals_completed),
            "goals_completed": int(self.goals_completed),
            "dist_to_goal": float(dist_to_wp),
            "goal_arrival_radius": float(self.goal_arrival_radius) if self.arena_mode else 1.0,
            "goal_pos": self.goal_pos.tolist() if self.arena_mode else list(COURSE_WAYPOINTS[self.current_wp_idx][0]),
            "target_local_pos": (torso_xmat.T @ (target_pos - torso_xpos)).tolist(),
            "gaze_alignment": float(cos_gaze),
            "stamina": float(self.stamina),
            "health": float(self.health),
            "hunger": float(self.hunger),
            "step": self.current_step,
            "lidar_grid": self.current_lidar_grid.tolist(),
            "in_leeway": bool(in_leeway) if self.arena_mode else False,
            "leeway_remaining": int(self.goal_transition_steps) if self.arena_mode else 0,
            "stream_mode": "cruise",
            "active_reach_hand": "none",
            "reach_touches": 0,
            "getup_success": False,
            "stand_still_score": 0.0,
        }

        return self._get_obs(), float(total_reward), terminated, truncated, info

    def render_frame(
        self,
        width: int = 480,
        height: int = 320,
        camera_name: str = "behind_cam",
        ghost_datas: Optional[List[Any]] = None,
        ghost_alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Renders an RGB numpy image of the current simulation frame.
        Optionally injects translucent ghost avatars into mjvScene for parallel multi-agent swarm visualization.
        """
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            
        self._renderer.update_scene(self.data, camera=camera_name)

        # Inject dynamic course waypoint goal markers into mjvScene
        scene = self._renderer.scene
        identity_mat = np.eye(3).flatten()
        if self.stream_mode == "reach":
            # Stream 3 Reach: Glowing magenta hand target orb + vertical guide line
            rx, ry, rz = float(self.reach_target_pos[0]), float(self.reach_target_pos[1]), float(self.reach_target_pos[2])
            if scene.ngeom < scene.maxgeom:
                dst_orb = scene.geoms[scene.ngeom]
                orb_pos = np.array([rx, ry, rz], dtype=np.float64)
                orb_size = np.array([0.09, 0.09, 0.09], dtype=np.float64)
                orb_rgba = np.array([0.88, 0.20, 0.95, 0.95], dtype=np.float32)  # Radiant Magenta
                mujoco.mjv_initGeom(dst_orb, mujoco.mjtGeom.mjGEOM_SPHERE, orb_size, orb_pos, identity_mat, orb_rgba)
                dst_orb.transparent = 1
                scene.ngeom += 1

            if scene.ngeom < scene.maxgeom:
                dst_guide = scene.geoms[scene.ngeom]
                guide_pos = np.array([rx, ry, max(0.01, rz / 2.0)], dtype=np.float64)
                guide_size = np.array([0.012, max(0.01, rz / 2.0), 0.0], dtype=np.float64)
                guide_rgba = np.array([0.88, 0.20, 0.95, 0.35], dtype=np.float32)
                mujoco.mjv_initGeom(dst_guide, mujoco.mjtGeom.mjGEOM_CYLINDER, guide_size, guide_pos, identity_mat, guide_rgba)
                dst_guide.transparent = 1
                scene.ngeom += 1

        elif self.stream_mode == "getup":
            # Stream 2 Get-Up: Golden upright elevation target disc
            torso_xpos = self.data.xpos[self.torso_body_id]
            tx, ty = float(torso_xpos[0]), float(torso_xpos[1])
            if scene.ngeom < scene.maxgeom:
                dst_target = scene.geoms[scene.ngeom]
                target_pos = np.array([tx, ty, 1.33], dtype=np.float64)
                target_size = np.array([0.25, 0.02, 0.0], dtype=np.float64)
                target_rgba = np.array([0.98, 0.75, 0.15, 0.65], dtype=np.float32)  # Glowing Amber
                mujoco.mjv_initGeom(dst_target, mujoco.mjtGeom.mjGEOM_CYLINDER, target_size, target_pos, identity_mat, target_rgba)
                dst_target.transparent = 1
                scene.ngeom += 1

        elif self.stream_mode == "stand":
            # Stream 1 Stand: Cyan stillness equilibrium ground pad
            if scene.ngeom < scene.maxgeom:
                dst_ring = scene.geoms[scene.ngeom]
                ring_pos = np.array([0.0, 0.0, 0.015], dtype=np.float64)
                ring_size = np.array([0.35, 0.015, 0.0], dtype=np.float64)
                ring_rgba = np.array([0.22, 0.74, 0.97, 0.50], dtype=np.float32)  # Cyan Pad
                mujoco.mjv_initGeom(dst_ring, mujoco.mjtGeom.mjGEOM_CYLINDER, ring_size, ring_pos, identity_mat, ring_rgba)
                dst_ring.transparent = 1
                scene.ngeom += 1

        elif self.arena_mode:
            # ACTIVE ARENA GOAL: Glowing floating target orb + vertical laser beacon column + ground target ring
            gx, gy, gz = float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2])
            
            # 1. Floating target orb (sphere)
            if scene.ngeom < scene.maxgeom:
                dst_orb = scene.geoms[scene.ngeom]
                orb_pos = np.array([gx, gy, gz + 0.35], dtype=np.float64)
                orb_size = np.array([0.22, 0.22, 0.22], dtype=np.float64)
                orb_rgba = np.array([0.98, 0.75, 0.15, 0.90], dtype=np.float32)  # Vibrant Gold
                mujoco.mjv_initGeom(dst_orb, mujoco.mjtGeom.mjGEOM_SPHERE, orb_size, orb_pos, identity_mat, orb_rgba)
                dst_orb.transparent = 1
                scene.ngeom += 1

            # 2. Vertical laser beacon column (cylinder)
            if scene.ngeom < scene.maxgeom:
                dst_beam = scene.geoms[scene.ngeom]
                beam_pos = np.array([gx, gy, 1.25], dtype=np.float64)
                beam_size = np.array([0.05, 1.25, 0.0], dtype=np.float64)
                beam_rgba = np.array([0.22, 0.74, 0.97, 0.55], dtype=np.float32)  # Glowing Cyan
                mujoco.mjv_initGeom(dst_beam, mujoco.mjtGeom.mjGEOM_CYLINDER, beam_size, beam_pos, identity_mat, beam_rgba)
                dst_beam.transparent = 1
                scene.ngeom += 1

            # 3. Ground target arrival ring
            if scene.ngeom < scene.maxgeom:
                dst_ring = scene.geoms[scene.ngeom]
                ring_pos = np.array([gx, gy, 0.02], dtype=np.float64)
                ring_size = np.array([self.goal_arrival_radius, 0.02, 0.0], dtype=np.float64)
                ring_rgba = np.array([0.22, 0.74, 0.97, 0.45], dtype=np.float32)
                mujoco.mjv_initGeom(dst_ring, mujoco.mjtGeom.mjGEOM_CYLINDER, ring_size, ring_pos, identity_mat, ring_rgba)
                dst_ring.transparent = 1
                scene.ngeom += 1
        else:
            for wp_idx, (wp_coord, wp_name, wp_radius) in enumerate(COURSE_WAYPOINTS):
                if scene.ngeom + 3 >= scene.maxgeom:
                    break
                wx = float(wp_coord[0])
                wy = float(wp_coord[1])
                wz = float(wp_coord[2])

                if wp_idx == self.current_wp_idx:
                    # ACTIVE GOAL: Glowing floating target orb + vertical laser beacon column + ground target ring
                    # 1. Floating target orb (sphere)
                    dst_orb = scene.geoms[scene.ngeom]
                    orb_pos = np.array([wx, wy, wz + 0.35], dtype=np.float64)
                    orb_size = np.array([0.18, 0.18, 0.18], dtype=np.float64)
                    orb_rgba = np.array([0.98, 0.75, 0.15, 0.90], dtype=np.float32) # Vibrant Gold
                    mujoco.mjv_initGeom(dst_orb, mujoco.mjtGeom.mjGEOM_SPHERE, orb_size, orb_pos, identity_mat, orb_rgba)
                    dst_orb.transparent = 1
                    scene.ngeom += 1

                    # 2. Vertical laser beacon column (cylinder)
                    if scene.ngeom < scene.maxgeom:
                        dst_beam = scene.geoms[scene.ngeom]
                        beam_pos = np.array([wx, wy, wz + 0.60], dtype=np.float64)
                        beam_size = np.array([0.05, 0.60, 0.0], dtype=np.float64)
                        beam_rgba = np.array([0.22, 0.74, 0.97, 0.55], dtype=np.float32) # Glowing Cyan
                        mujoco.mjv_initGeom(dst_beam, mujoco.mjtGeom.mjGEOM_CYLINDER, beam_size, beam_pos, identity_mat, beam_rgba)
                        dst_beam.transparent = 1
                        scene.ngeom += 1

                    # 3. Ground target landing pad ring
                    if scene.ngeom < scene.maxgeom:
                        dst_ring = scene.geoms[scene.ngeom]
                        ring_pos = np.array([wx, wy, max(0.01, wz - 0.05)], dtype=np.float64)
                        ring_size = np.array([float(wp_radius) * 0.6, 0.02, 0.0], dtype=np.float64)
                        ring_rgba = np.array([0.22, 0.74, 0.97, 0.45], dtype=np.float32)
                        mujoco.mjv_initGeom(dst_ring, mujoco.mjtGeom.mjGEOM_CYLINDER, ring_size, ring_pos, identity_mat, ring_rgba)
                        dst_ring.transparent = 1
                        scene.ngeom += 1

                elif wp_idx < self.current_wp_idx:
                    # COMPLETED GOAL: Subtle emerald disc on surface
                    dst_comp = scene.geoms[scene.ngeom]
                    comp_pos = np.array([wx, wy, max(0.01, wz - 0.05)], dtype=np.float64)
                    comp_size = np.array([0.25, 0.015, 0.0], dtype=np.float64)
                    comp_rgba = np.array([0.10, 0.85, 0.35, 0.40], dtype=np.float32) # Emerald Green
                    mujoco.mjv_initGeom(dst_comp, mujoco.mjtGeom.mjGEOM_CYLINDER, comp_size, comp_pos, identity_mat, comp_rgba)
                    dst_comp.transparent = 1
                    scene.ngeom += 1

                else:
                    # UPCOMING GOAL: Translucent waypoint marker disc
                    dst_up = scene.geoms[scene.ngeom]
                    up_pos = np.array([wx, wy, max(0.01, wz - 0.05)], dtype=np.float64)
                    up_size = np.array([0.16, 0.015, 0.0], dtype=np.float64)
                    up_rgba = np.array([0.60, 0.65, 0.80, 0.25], dtype=np.float32) # Translucent Silver
                    mujoco.mjv_initGeom(dst_up, mujoco.mjtGeom.mjGEOM_CYLINDER, up_size, up_pos, identity_mat, up_rgba)
                    dst_up.transparent = 1
                    scene.ngeom += 1

        # Inject translucent ghost avatars if provided
        if ghost_datas:
            scene = self._renderer.scene
            for g_idx, g_data in enumerate(ghost_datas):
                color = GHOST_PALETTE[g_idx % len(GHOST_PALETTE)]
                if ghost_alpha is not None:
                    color = color.copy()
                    color[3] = float(ghost_alpha)
                for gid in self.humanoid_geoms:
                    if scene.ngeom >= scene.maxgeom:
                        break
                    dst = scene.geoms[scene.ngeom]
                    pos = g_data.geom_xpos[gid].astype(np.float64)
                    size = self.model.geom_size[gid].astype(np.float64)
                    mat = g_data.geom_xmat[gid].astype(np.float64)
                    geom_type = int(self.model.geom_type[gid])

                    mujoco.mjv_initGeom(dst, geom_type, size, pos, mat, color)
                    dst.transparent = 1
                    scene.ngeom += 1

        return self._renderer.render()
