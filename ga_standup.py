from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


DEFAULT_STEPS = 480
SIM_DT = 1.0 / 240.0
JOINT_FORCE_LIMIT = 46.0
POPULATION_SPACING = 1.4
DEFAULT_POPULATION = 100
DEFAULT_PARENT_POOL = 10
DEFAULT_KEYFRAMES = 5
START_BASE_HEIGHT = 0.25
SETTLING_STEPS = 60
X_AXIS_CAPSULE_ORIENTATION = p.getQuaternionFromEuler([0, math.radians(90), 0])
FOOT_HALF_EXTENTS = [0.055, 0.18, 0.14]
PLANE_GROUP = 1
ROBOT_GROUP = 2
STABLE_TAIL_FRACTION = 0.20

SPINE = 0
NECK = 1
LEFT_SHOULDER = 2
LEFT_ELBOW = 3
RIGHT_SHOULDER = 4
RIGHT_ELBOW = 5
LEFT_HIP = 6
LEFT_KNEE = 7
LEFT_ANKLE = 8
RIGHT_HIP = 9
RIGHT_KNEE = 10
RIGHT_ANKLE = 11
LINK_COUNT = 12
CONTROLLED_JOINTS = (
    SPINE,
    NECK,
    LEFT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_HIP,
    RIGHT_KNEE,
)
CONTROL_DOF = len(CONTROLLED_JOINTS)

JOINT_LOWERS = np.array(
    [-0.50, -0.50, -2.00, 0.00, -2.00, 0.00, -1.50, -2.50, -1.50, -2.50],
    dtype=float,
)
JOINT_UPPERS = np.array(
    [1.50, 0.50, 2.00, 2.50, 2.00, 2.50, 1.50, 0.00, 1.50, 0.00],
    dtype=float,
)


@dataclass(frozen=True)
class ShapeBundle:
    pelvis_col: int
    torso_col: int
    head_col: int
    upper_arm_col: int
    forearm_col: int
    thigh_col: int
    shin_col: int
    foot_col: int
    pelvis_vis: int
    torso_vis: int
    head_vis: int
    upper_arm_vis: int
    forearm_vis: int
    thigh_vis: int
    shin_vis: int
    foot_vis: int


@dataclass
class Genome:
    """Free keyframe controller for an emergent get-up strategy."""

    keyframes: np.ndarray  # shape: (num_keyframes, CONTROL_DOF)
    durations: np.ndarray  # shape: (num_keyframes,)
    force_scales: np.ndarray  # shape: (num_keyframes,)

    @classmethod
    def random(cls, num_keyframes: int = DEFAULT_KEYFRAMES) -> "Genome":
        keyframes = np.random.uniform(
            JOINT_LOWERS,
            JOINT_UPPERS,
            size=(num_keyframes, CONTROL_DOF),
        )
        durations = np.random.uniform(0.08, 0.35, size=num_keyframes)
        force_scales = np.random.uniform(0.45, 1.55, size=num_keyframes)
        return cls(keyframes=keyframes, durations=durations, force_scales=force_scales)

    @property
    def num_keyframes(self) -> int:
        return int(self.keyframes.shape[0])

    def phase_at(self, step: int, total_steps: int) -> tuple[int, float]:
        progress = step / max(1, total_steps - 1)
        weights = self.durations / max(0.0001, float(np.sum(self.durations)))
        cumulative = np.cumsum(weights)
        phase_index = int(np.searchsorted(cumulative, progress, side="left"))
        phase_index = min(phase_index, self.num_keyframes - 1)
        phase_start = 0.0 if phase_index == 0 else float(cumulative[phase_index - 1])
        phase_end = float(cumulative[phase_index])
        alpha = (progress - phase_start) / max(0.0001, phase_end - phase_start)
        return phase_index, float(np.clip(alpha, 0.0, 1.0))

    def control_at(self, step: int, total_steps: int) -> tuple[np.ndarray, float, int]:
        phase_index, alpha = self.phase_at(step, total_steps)
        current = self.keyframes[phase_index]
        previous = (
            np.zeros(CONTROL_DOF, dtype=float)
            if phase_index == 0
            else self.keyframes[phase_index - 1]
        )

        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        target = (1.0 - eased) * previous + eased * current
        force_scale = float(self.force_scales[phase_index])
        return target, force_scale, phase_index


@dataclass
class EvaluationStats:
    average_head_height: float
    final_head_height: float
    final_head_over_pelvis: float
    tail_head_over_pelvis: float
    tail_torso_upright: float
    tail_both_feet: float
    tail_speed: float
    tail_stable_fraction: float
    drift: float
    energy: float


def connect(gui: bool) -> int:
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=35,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.65],
        )
    return client


def fit_population_camera(robot_count: int) -> None:
    if not p.isConnected() or robot_count <= 1:
        return

    p.resetDebugVisualizerCamera(
        cameraDistance=max(4.0, robot_count * POPULATION_SPACING * 0.38),
        cameraYaw=35,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.65],
    )


def safe_disconnect() -> None:
    if p.isConnected():
        p.disconnect()


def make_box_collision(half_extents: list[float]) -> int:
    return p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)


def make_box_visual(half_extents: list[float], color: list[float]) -> int:
    return p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)


def make_capsule_collision(radius: float, height: float) -> int:
    return p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)


def make_capsule_visual(radius: float, height: float, color: list[float]) -> int:
    return p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
        length=height,
        rgbaColor=color,
    )


def make_x_capsule_collision(radius: float, height: float) -> int:
    return p.createCollisionShape(
        p.GEOM_CAPSULE,
        radius=radius,
        height=height,
        collisionFrameOrientation=X_AXIS_CAPSULE_ORIENTATION,
    )


def make_x_capsule_visual(radius: float, height: float, color: list[float]) -> int:
    return p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
        length=height,
        rgbaColor=color,
        visualFrameOrientation=X_AXIS_CAPSULE_ORIENTATION,
    )


def make_sphere_collision(radius: float) -> int:
    return p.createCollisionShape(p.GEOM_SPHERE, radius=radius)


def make_sphere_visual(radius: float, color: list[float]) -> int:
    return p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)


def build_shared_shapes() -> ShapeBundle:
    body_color = [0.82, 0.72, 0.43, 1]
    limb_color = [0.88, 0.78, 0.48, 1]
    foot_color = [0.55, 0.48, 0.27, 1]

    return ShapeBundle(
        pelvis_col=make_sphere_collision(0.16),
        torso_col=make_x_capsule_collision(0.15, 0.36),
        head_col=make_sphere_collision(0.115),
        upper_arm_col=make_x_capsule_collision(0.055, 0.22),
        forearm_col=make_x_capsule_collision(0.045, 0.22),
        thigh_col=make_x_capsule_collision(0.07, 0.32),
        shin_col=make_x_capsule_collision(0.06, 0.34),
        foot_col=make_box_collision(FOOT_HALF_EXTENTS),
        pelvis_vis=make_sphere_visual(0.16, body_color),
        torso_vis=make_x_capsule_visual(0.15, 0.36, body_color),
        head_vis=make_sphere_visual(0.115, body_color),
        upper_arm_vis=make_x_capsule_visual(0.055, 0.22, limb_color),
        forearm_vis=make_x_capsule_visual(0.045, 0.22, limb_color),
        thigh_vis=make_x_capsule_visual(0.07, 0.32, limb_color),
        shin_vis=make_x_capsule_visual(0.06, 0.34, limb_color),
        foot_vis=make_box_visual(FOOT_HALF_EXTENTS, foot_color),
    )


def create_humanoid_robot(
    origin_y: float = 0.0,
    shapes: ShapeBundle | None = None,
) -> tuple[int, list[int]]:
    """Create a compact humanoid already lying supine on the ground.

    The body is modeled directly in world-like coordinates: X is head/feet,
    Z is up. The hip and knee joints rotate around Z so the legs fold sideways
    across the floor. Feet are fixed pads, not controllable ankle joints.
    """

    if shapes is None:
        shapes = build_shared_shapes()

    body_id = p.createMultiBody(
        baseMass=1.4,
        baseCollisionShapeIndex=shapes.pelvis_col,
        baseVisualShapeIndex=shapes.pelvis_vis,
        basePosition=[0, origin_y, START_BASE_HEIGHT],
        baseOrientation=[0, 0, 0, 1],
        linkMasses=[1.2, 0.45, 0.35, 0.25, 0.35, 0.25, 0.75, 0.55, 0.25, 0.75, 0.55, 0.25],
        linkCollisionShapeIndices=[
            shapes.torso_col,
            shapes.head_col,
            shapes.upper_arm_col,
            shapes.forearm_col,
            shapes.upper_arm_col,
            shapes.forearm_col,
            shapes.thigh_col,
            shapes.shin_col,
            shapes.foot_col,
            shapes.thigh_col,
            shapes.shin_col,
            shapes.foot_col,
        ],
        linkVisualShapeIndices=[
            shapes.torso_vis,
            shapes.head_vis,
            shapes.upper_arm_vis,
            shapes.forearm_vis,
            shapes.upper_arm_vis,
            shapes.forearm_vis,
            shapes.thigh_vis,
            shapes.shin_vis,
            shapes.foot_vis,
            shapes.thigh_vis,
            shapes.shin_vis,
            shapes.foot_vis,
        ],
        linkPositions=[
            [0.30, 0, 0.02],
            [0.36, 0, 0.02],
            [0.10, -0.24, 0.02],
            [0.24, 0, 0],
            [0.10, 0.24, 0.02],
            [0.24, 0, 0],
            [-0.24, -0.09, 0],
            [-0.34, 0, 0],
            [-0.27, 0, 0.095],
            [-0.24, 0.09, 0],
            [-0.34, 0, 0],
            [-0.27, 0, 0.095],
        ],
        linkOrientations=[[0, 0, 0, 1]] * LINK_COUNT,
        linkInertialFramePositions=[[0, 0, 0]] * LINK_COUNT,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * LINK_COUNT,
        linkParentIndices=[0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11],
        linkJointTypes=[
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_FIXED,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_FIXED,
        ],
        linkJointAxis=[
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
        ],
    )

    controlled = []
    p.setCollisionFilterGroupMask(body_id, -1, ROBOT_GROUP, PLANE_GROUP)
    for joint_id in range(p.getNumJoints(body_id)):
        p.setCollisionFilterGroupMask(body_id, joint_id, ROBOT_GROUP, PLANE_GROUP)
        p.changeDynamics(body_id, joint_id, lateralFriction=1.4, spinningFriction=0.1)
        p.setJointMotorControl2(body_id, joint_id, p.VELOCITY_CONTROL, force=0)
        if joint_id in CONTROLLED_JOINTS:
            controlled.append(joint_id)
            control_index = CONTROLLED_JOINTS.index(joint_id)
            p.changeDynamics(
                body_id,
                joint_id,
                jointLowerLimit=float(JOINT_LOWERS[control_index]),
                jointUpperLimit=float(JOINT_UPPERS[control_index]),
            )

    p.changeDynamics(body_id, -1, lateralFriction=1.4, spinningFriction=0.1)
    return body_id, controlled


def reset_world(robot_count: int = 1) -> list[tuple[int, list[int], float]]:
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)
    plane_id = p.loadURDF("plane.urdf")
    p.setCollisionFilterGroupMask(plane_id, -1, PLANE_GROUP, ROBOT_GROUP)

    center = (robot_count - 1) * 0.5
    shapes = build_shared_shapes()
    robots = []
    for index in range(robot_count):
        origin_y = (index - center) * POPULATION_SPACING
        robot_id, joints = create_humanoid_robot(origin_y, shapes)
        robots.append((robot_id, joints, origin_y))

    p.performCollisionDetection()
    return robots


def apply_target_pose(
    robot_id: int,
    joints: list[int],
    target: np.ndarray,
    force_scale: float,
) -> None:
    for joint_id, target_position in zip(joints, target):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(target_position),
            force=JOINT_FORCE_LIMIT * force_scale,
            positionGain=0.15,
            velocityGain=0.40,
            maxVelocity=4.0,
        )


def feet_on_world(robot_id: int) -> tuple[bool, bool]:
    left_foot_contacts = p.getContactPoints(bodyA=robot_id, linkIndexA=LEFT_ANKLE)
    right_foot_contacts = p.getContactPoints(bodyA=robot_id, linkIndexA=RIGHT_ANKLE)
    left_foot_on_world = any(contact[2] != robot_id for contact in left_foot_contacts)
    right_foot_on_world = any(contact[2] != robot_id for contact in right_foot_contacts)
    return left_foot_on_world, right_foot_on_world


def torso_forward_z(robot_id: int) -> float:
    torso_orn = p.getLinkState(robot_id, SPINE)[1]
    matrix = p.getMatrixFromQuaternion(torso_orn)
    return float(matrix[6])


def posture_sample(robot_id: int, origin_y: float) -> dict[str, float]:
    base_pos, _base_orn = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
    head_pos = p.getLinkState(robot_id, NECK)[0]
    left_foot_on_world, right_foot_on_world = feet_on_world(robot_id)
    speed = float(np.linalg.norm(linear_velocity) + 0.25 * np.linalg.norm(angular_velocity))
    head_over_pelvis = float(head_pos[2] - base_pos[2])
    torso_upright = max(0.0, torso_forward_z(robot_id))
    both_feet = float(left_foot_on_world and right_foot_on_world)
    stable = float(
        both_feet
        and speed < 0.28
        and head_over_pelvis > 0.34
        and torso_upright > 0.45
    )

    return {
        "head_height": float(head_pos[2]),
        "head_over_pelvis": head_over_pelvis,
        "torso_upright": torso_upright,
        "both_feet": both_feet,
        "speed": speed,
        "stable": stable,
        "drift": float(abs(base_pos[0]) + abs(base_pos[1] - origin_y)),
    }


def fitness(stats: EvaluationStats, mode: str) -> float:
    if mode == "simple":
        return stats.average_head_height + 0.20 * stats.final_head_height

    completion_bonus = 2.5 * stats.tail_stable_fraction
    if (
        stats.tail_both_feet > 0.80
        and stats.tail_speed < 0.28
        and stats.tail_head_over_pelvis > 0.34
        and stats.tail_torso_upright > 0.45
    ):
        completion_bonus += 3.0

    return (
        0.08 * stats.average_head_height
        + 0.30 * stats.final_head_height
        + 0.75 * stats.final_head_over_pelvis
        + 0.70 * stats.tail_head_over_pelvis
        + 0.60 * stats.tail_torso_upright
        + 0.45 * stats.tail_both_feet
        + completion_bonus
        - 0.20 * stats.tail_speed
        - 0.06 * stats.drift
        - 0.00001 * stats.energy
    )


def evaluate_population(
    population: list[Genome],
    steps: int,
    fitness_mode: str,
    gui: bool = False,
    generation: int | None = None,
) -> list[float]:
    robots = reset_world(len(population))
    if gui:
        fit_population_camera(len(population))

    for _ in range(SETTLING_STEPS):
        if not p.isConnected():
            break
        p.stepSimulation()
        if gui:
            time.sleep(SIM_DT)

    for index, (robot_id, joints, _origin_y) in enumerate(robots):
        base_pos, _base_orn = p.getBasePositionAndOrientation(robot_id)
        robots[index] = (robot_id, joints, float(base_pos[1]))

    energies = np.zeros(len(population))
    head_height_sums = np.zeros(len(population))
    tail_head_over_pelvis_sums = np.zeros(len(population))
    tail_torso_upright_sums = np.zeros(len(population))
    tail_both_feet_sums = np.zeros(len(population))
    tail_speed_sums = np.zeros(len(population))
    tail_stable_sums = np.zeros(len(population))
    final_samples: list[dict[str, float] | None] = [None for _ in population]
    debug_text_ids = [-1 for _ in population]
    tail_start = max(0, int(steps * (1.0 - STABLE_TAIL_FRACTION)))
    tail_count = 0
    simulation_alive = True

    for step in range(steps):
        if not p.isConnected():
            break

        for index, genome in enumerate(population):
            robot_id, joints, _origin_y = robots[index]
            target, force_scale, _phase_index = genome.control_at(step, steps)
            try:
                apply_target_pose(robot_id, joints, target, force_scale)
            except p.error:
                simulation_alive = False
                break

        if not simulation_alive:
            break

        p.stepSimulation()

        if not p.isConnected():
            break

        for index, (robot_id, joints, _origin_y) in enumerate(robots):
            try:
                sample = posture_sample(robot_id, robots[index][2])
            except p.error:
                simulation_alive = False
                break
            final_samples[index] = sample
            head_height_sums[index] += sample["head_height"]
            if step >= tail_start:
                tail_head_over_pelvis_sums[index] += sample["head_over_pelvis"]
                tail_torso_upright_sums[index] += sample["torso_upright"]
                tail_both_feet_sums[index] += sample["both_feet"]
                tail_speed_sums[index] += sample["speed"]
                tail_stable_sums[index] += sample["stable"]
            for joint_id in joints:
                try:
                    _position, velocity, _reaction_forces, motor_torque = p.getJointState(
                        robot_id,
                        joint_id,
                    )
                except p.error:
                    simulation_alive = False
                    break
                energies[index] += abs(motor_torque * velocity) * SIM_DT
            if not simulation_alive:
                break

        if not simulation_alive:
            break
        if step >= tail_start:
            tail_count += 1

        if gui and p.isConnected():
            for index, (robot_id, _joints, origin_y) in enumerate(robots):
                _target, _force_scale, phase_index = population[index].control_at(step, steps)
                label = f"g{generation} #{index} keyframe={phase_index}"
                try:
                    debug_text_ids[index] = p.addUserDebugText(
                        label,
                        [-0.9, origin_y - 0.25, 1.25],
                        textColorRGB=[0.05, 0.05, 0.05],
                        textSize=1.0,
                        replaceItemUniqueId=debug_text_ids[index],
                    )
                except p.error:
                    break
            time.sleep(SIM_DT)

    completed_steps = max(1, step + 1)
    average_head_heights = head_height_sums / completed_steps
    tail_count = max(1, tail_count)
    if not p.isConnected():
        return (average_head_heights - energies * 0.00002).tolist()

    scores = []
    for index, (robot_id, _joints, origin_y) in enumerate(robots):
        sample = final_samples[index] or posture_sample(robot_id, origin_y)
        stats = EvaluationStats(
            average_head_height=float(average_head_heights[index]),
            final_head_height=float(sample["head_height"]),
            final_head_over_pelvis=float(sample["head_over_pelvis"]),
            tail_head_over_pelvis=float(tail_head_over_pelvis_sums[index] / tail_count),
            tail_torso_upright=float(tail_torso_upright_sums[index] / tail_count),
            tail_both_feet=float(tail_both_feet_sums[index] / tail_count),
            tail_speed=float(tail_speed_sums[index] / tail_count),
            tail_stable_fraction=float(tail_stable_sums[index] / tail_count),
            drift=float(sample["drift"]),
            energy=float(energies[index]),
        )
        scores.append(fitness(stats, fitness_mode))
    return scores


def tournament(population: list[tuple[float, Genome]], size: int = 3) -> Genome:
    contenders = random.sample(population, k=min(size, len(population)))
    contenders.sort(key=lambda item: item[0], reverse=True)
    return contenders[0][1]


def crossover(a: Genome, b: Genome) -> Genome:
    keyframe_mask = np.random.random(a.keyframes.shape) < 0.5
    duration_mask = np.random.random(a.durations.shape) < 0.5
    force_mask = np.random.random(a.force_scales.shape) < 0.5
    return Genome(
        keyframes=np.where(keyframe_mask, a.keyframes, b.keyframes).copy(),
        durations=np.where(duration_mask, a.durations, b.durations).copy(),
        force_scales=np.where(force_mask, a.force_scales, b.force_scales).copy(),
    )


def mutate(genome: Genome, rate: float, scale: float) -> Genome:
    keyframes = genome.keyframes.copy()
    keyframe_mask = np.random.random(keyframes.shape) < rate
    keyframes[keyframe_mask] += np.random.normal(
        0.0,
        scale,
        size=keyframes[keyframe_mask].shape,
    )

    durations = genome.durations.copy()
    duration_mask = np.random.random(durations.shape) < rate
    durations[duration_mask] += np.random.normal(
        0.0,
        0.35 * scale,
        size=durations[duration_mask].shape,
    )

    force_scales = genome.force_scales.copy()
    force_mask = np.random.random(force_scales.shape) < rate
    force_scales[force_mask] += np.random.normal(
        0.0,
        0.45 * scale,
        size=force_scales[force_mask].shape,
    )

    # Rarely swap whole keyframes so evolution can discover different phase orders.
    if random.random() < rate:
        swap_a, swap_b = random.sample(range(genome.num_keyframes), k=2)
        keyframes[[swap_a, swap_b]] = keyframes[[swap_b, swap_a]]
        durations[[swap_a, swap_b]] = durations[[swap_b, swap_a]]
        force_scales[[swap_a, swap_b]] = force_scales[[swap_b, swap_a]]

    return Genome(
        keyframes=np.clip(keyframes, JOINT_LOWERS, JOINT_UPPERS),
        durations=np.clip(durations, 0.05, 0.60),
        force_scales=np.clip(force_scales, 0.20, 2.00),
    )


def next_generation(
    scored: list[tuple[float, Genome]],
    population_size: int,
    elite_count: int,
    parent_pool_size: int,
    mutation_rate: float,
    mutation_scale: float,
) -> list[Genome]:
    scored.sort(key=lambda item: item[0], reverse=True)
    parent_pool = scored[: min(parent_pool_size, len(scored))]
    elite_count = min(elite_count, population_size, len(scored))
    next_pop = [
        Genome(
            keyframes=g.keyframes.copy(),
            durations=g.durations.copy(),
            force_scales=g.force_scales.copy(),
        )
        for _, g in scored[:elite_count]
    ]

    while len(next_pop) < population_size:
        parent_a = tournament(parent_pool)
        parent_b = tournament(parent_pool)
        child = crossover(parent_a, parent_b)
        next_pop.append(mutate(child, mutation_rate, mutation_scale))

    return next_pop


def train(args: argparse.Namespace) -> Genome:
    population = [Genome.random(args.keyframes) for _ in range(args.population)]
    best: tuple[float, Genome] | None = None

    connect(gui=args.watch_population)
    try:
        for generation in range(args.generations):
            scores = evaluate_population(
                population,
                args.steps,
                fitness_mode=args.fitness,
                gui=args.watch_population,
                generation=generation,
            )
            scored = list(zip(scores, population))
            scored.sort(key=lambda item: item[0], reverse=True)
            best = scored[0] if best is None or scored[0][0] > best[0] else best

            print(
                f"gen={generation:03d} "
                f"best={scored[0][0]:8.3f} "
                f"global_best={best[0]:8.3f} "
                f"avg={np.mean(scores):8.3f} "
                f"fitness={args.fitness} "
                f"keyframes={scored[0][1].num_keyframes}"
            )

            if args.watch_population and not p.isConnected():
                break

            population = next_generation(
                scored,
                population_size=args.population,
                elite_count=args.elites,
                parent_pool_size=args.parent_pool,
                mutation_rate=args.mutation_rate,
                mutation_scale=args.mutation_scale,
            )
    finally:
        safe_disconnect()

    assert best is not None
    return best[1]


def replay(genome: Genome, args: argparse.Namespace) -> None:
    connect(gui=True)
    try:
        replay_count = 0
        while p.isConnected():
            score = evaluate_population(
                [genome],
                args.steps,
                fitness_mode=args.fitness,
                gui=True,
            )[0]
            print(f"replay score={score:.3f}")
            replay_count += 1
            if args.replay_count and replay_count >= args.replay_count:
                break
            time.sleep(0.8)
    except KeyboardInterrupt:
        pass
    finally:
        safe_disconnect()


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be greater than 0")
        return parsed

    def at_least_two(value: str) -> int:
        parsed = int(value)
        if parsed < 2:
            raise argparse.ArgumentTypeError("must be 2 or greater")
        return parsed

    def non_negative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be 0 or greater")
        return parsed

    def non_negative_float(value: str) -> float:
        parsed = float(value)
        if parsed < 0.0:
            raise argparse.ArgumentTypeError("must be 0 or greater")
        return parsed

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=positive_int, default=30)
    parser.add_argument("--population", type=positive_int, default=DEFAULT_POPULATION)
    parser.add_argument("--elites", type=non_negative_int, default=5)
    parser.add_argument("--parent-pool", type=positive_int, default=DEFAULT_PARENT_POOL)
    parser.add_argument("--steps", type=positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--keyframes", type=at_least_two, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--fitness", choices=("simple", "stable"), default="simple")
    parser.add_argument("--mutation-rate", type=non_negative_float, default=0.12)
    parser.add_argument("--mutation-scale", type=non_negative_float, default=0.22)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--watch-population", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-count", type=non_negative_int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    best = train(args)
    if args.replay:
        replay(best, args)


if __name__ == "__main__":
    main()
