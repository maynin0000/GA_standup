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
CONTROL_JOINTS = 12


@dataclass(frozen=True)
class Primitive:
    name: str
    targets: np.ndarray
    duration: float


def pose(
    spine: float,
    neck: float,
    left_shoulder: float,
    left_elbow: float,
    right_shoulder: float,
    right_elbow: float,
    left_hip: float,
    left_knee: float,
    left_ankle: float,
    right_hip: float,
    right_knee: float,
    right_ankle: float,
) -> np.ndarray:
    return np.array(
        [
            spine,
            neck,
            left_shoulder,
            left_elbow,
            right_shoulder,
            right_elbow,
            left_hip,
            left_knee,
            left_ankle,
            right_hip,
            right_knee,
            right_ankle,
        ],
        dtype=float,
    )


MOVEMENT_PRIMITIVES = [
    Primitive(
        "sit_up",
        pose(0.95, -0.2, -0.9, 0.7, -0.9, 0.7, 0.45, -0.85, 0.25, 0.45, -0.85, 0.25),
        0.16,
    ),
    Primitive(
        "fold_right_leg",
        pose(1.05, -0.15, -0.7, 0.55, -0.7, 0.55, 0.35, -0.75, 0.2, 1.25, -1.45, 0.55),
        0.16,
    ),
    Primitive(
        "fold_left_leg",
        pose(1.05, -0.15, -0.65, 0.5, -0.65, 0.5, 1.25, -1.45, 0.55, 1.05, -1.15, 0.4),
        0.16,
    ),
    Primitive(
        "plant_feet",
        pose(0.85, -0.1, -0.35, 0.3, -0.35, 0.3, 0.85, -1.15, 0.65, 0.85, -1.15, 0.65),
        0.16,
    ),
    Primitive(
        "extend_legs",
        pose(0.35, 0.0, -0.15, 0.15, -0.15, 0.15, 0.25, -0.25, 0.05, 0.25, -0.25, 0.05),
        0.18,
    ),
    Primitive(
        "stabilize",
        pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        0.18,
    ),
]


@dataclass
class Genome:
    """Primitive order plus strength values for a get-up strategy."""

    weights: np.ndarray  # shape: (len(MOVEMENT_PRIMITIVES),)
    order: np.ndarray  # first four shuffled, final two fixed

    @classmethod
    def random(cls) -> "Genome":
        order = np.array(random.sample(range(4), k=4) + [4, 5])
        weights = np.random.uniform(0.75, 1.25, size=len(MOVEMENT_PRIMITIVES))
        return cls(weights=weights, order=order)

    def target_at(self, step: int, total_steps: int) -> np.ndarray:
        phase_index, alpha = primitive_phase(step, total_steps)
        primitive_index = int(self.order[phase_index])
        current = (
            MOVEMENT_PRIMITIVES[primitive_index].targets
            * self.weights[primitive_index]
        )

        if phase_index == 0:
            previous = current
        else:
            previous_index = int(self.order[phase_index - 1])
            previous = (
                MOVEMENT_PRIMITIVES[previous_index].targets
                * self.weights[previous_index]
            )

        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - eased) * previous + eased * current

    def primitive_name_at(self, step: int, total_steps: int) -> str:
        phase_index, _ = primitive_phase(step, total_steps)
        return MOVEMENT_PRIMITIVES[int(self.order[phase_index])].name


def primitive_phase(step: int, total_steps: int) -> tuple[int, float]:
    progress = step / max(1, total_steps - 1)
    start = 0.0

    for index, primitive in enumerate(MOVEMENT_PRIMITIVES):
        end = start + primitive.duration
        if progress <= end or index == len(MOVEMENT_PRIMITIVES) - 1:
            alpha = (progress - start) / max(0.0001, primitive.duration)
            return index, float(np.clip(alpha, 0.0, 1.0))
        start = end

    return len(MOVEMENT_PRIMITIVES) - 1, 1.0


def order_label(genome: Genome) -> str:
    return ">".join(MOVEMENT_PRIMITIVES[int(index)].name for index in genome.order)


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


def make_sphere_collision(radius: float) -> int:
    return p.createCollisionShape(p.GEOM_SPHERE, radius=radius)


def make_sphere_visual(radius: float, color: list[float]) -> int:
    return p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)


def create_humanoid_robot(origin_y: float = 0.0) -> tuple[int, list[int]]:
    """Create a compact humanoid: pelvis, torso, head, arms, legs, and feet."""

    body_color = [0.82, 0.72, 0.43, 1]
    limb_color = [0.88, 0.78, 0.48, 1]
    foot_color = [0.55, 0.48, 0.27, 1]

    pelvis_col = make_capsule_collision(0.12, 0.18)
    torso_col = make_capsule_collision(0.16, 0.34)
    head_col = make_sphere_collision(0.115)
    upper_arm_col = make_capsule_collision(0.055, 0.22)
    forearm_col = make_capsule_collision(0.045, 0.22)
    thigh_col = make_capsule_collision(0.07, 0.32)
    shin_col = make_capsule_collision(0.06, 0.34)
    foot_col = make_box_collision([0.13, 0.055, 0.035])

    pelvis_vis = make_capsule_visual(0.12, 0.18, body_color)
    torso_vis = make_capsule_visual(0.16, 0.34, body_color)
    head_vis = make_sphere_visual(0.115, body_color)
    upper_arm_vis = make_capsule_visual(0.055, 0.22, limb_color)
    forearm_vis = make_capsule_visual(0.045, 0.22, limb_color)
    thigh_vis = make_capsule_visual(0.07, 0.32, limb_color)
    shin_vis = make_capsule_visual(0.06, 0.34, limb_color)
    foot_vis = make_box_visual([0.13, 0.055, 0.035], foot_color)

    lying_orientation = p.getQuaternionFromEuler([0, math.radians(88), 0])

    body_id = p.createMultiBody(
        baseMass=1.4,
        baseCollisionShapeIndex=pelvis_col,
        baseVisualShapeIndex=pelvis_vis,
        basePosition=[0, origin_y, 0.28],
        baseOrientation=lying_orientation,
        linkMasses=[1.2, 0.45, 0.35, 0.25, 0.35, 0.25, 0.75, 0.55, 0.25, 0.75, 0.55, 0.25],
        linkCollisionShapeIndices=[
            torso_col,
            head_col,
            upper_arm_col,
            forearm_col,
            upper_arm_col,
            forearm_col,
            thigh_col,
            shin_col,
            foot_col,
            thigh_col,
            shin_col,
            foot_col,
        ],
        linkVisualShapeIndices=[
            torso_vis,
            head_vis,
            upper_arm_vis,
            forearm_vis,
            upper_arm_vis,
            forearm_vis,
            thigh_vis,
            shin_vis,
            foot_vis,
            thigh_vis,
            shin_vis,
            foot_vis,
        ],
        linkPositions=[
            [0, 0, 0.27],
            [0, 0, 0.35],
            [0, -0.2, 0.19],
            [0, 0, -0.24],
            [0, 0.2, 0.19],
            [0, 0, -0.24],
            [0, -0.08, -0.22],
            [0, 0, -0.36],
            [0.08, 0, -0.21],
            [0, 0.08, -0.22],
            [0, 0, -0.36],
            [0.08, 0, -0.21],
        ],
        linkOrientations=[[0, 0, 0, 1]] * CONTROL_JOINTS,
        linkInertialFramePositions=[[0, 0, 0]] * CONTROL_JOINTS,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * CONTROL_JOINTS,
        linkParentIndices=[0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11],
        linkJointTypes=[p.JOINT_REVOLUTE] * CONTROL_JOINTS,
        linkJointAxis=[[0, 1, 0]] * CONTROL_JOINTS,
    )

    controlled = []
    for joint_id in range(p.getNumJoints(body_id)):
        p.changeDynamics(body_id, joint_id, lateralFriction=1.4, spinningFriction=0.1)
        p.setJointMotorControl2(body_id, joint_id, p.VELOCITY_CONTROL, force=0)
        controlled.append(joint_id)

    p.changeDynamics(body_id, -1, lateralFriction=1.4, spinningFriction=0.1)
    return body_id, controlled


def reset_world(robot_count: int = 1) -> list[tuple[int, list[int], float]]:
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)
    p.loadURDF("plane.urdf")

    center = (robot_count - 1) * 0.5
    robots = []
    for index in range(robot_count):
        origin_y = (index - center) * POPULATION_SPACING
        robot_id, joints = create_humanoid_robot(origin_y)
        robots.append((robot_id, joints, origin_y))
    return robots


def apply_target_pose(robot_id: int, joints: list[int], target: np.ndarray) -> None:
    for joint_id, target_position in zip(joints, target):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(target_position),
            force=JOINT_FORCE_LIMIT,
            positionGain=0.55,
            velocityGain=0.12,
        )


def fitness(robot_id: int, origin_y: float, average_head_height: float, energy: float) -> float:
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, _yaw = p.getEulerFromQuaternion(base_orn)

    upright_bonus = max(0.0, 1.0 - abs(pitch) / math.pi) * 0.35
    roll_penalty = abs(roll) * 0.08
    drift_penalty = (abs(base_pos[0]) + abs(base_pos[1] - origin_y)) * 0.08
    energy_penalty = energy * 0.00002

    return average_head_height + upright_bonus - roll_penalty - drift_penalty - energy_penalty


def evaluate_population(
    population: list[Genome],
    steps: int,
    gui: bool = False,
    generation: int | None = None,
) -> list[float]:
    robots = reset_world(len(population))
    if gui:
        fit_population_camera(len(population))

    energies = np.zeros(len(population))
    head_height_sums = np.zeros(len(population))
    debug_text_ids = [-1 for _ in population]

    for step in range(steps):
        if not p.isConnected():
            break

        for index, genome in enumerate(population):
            robot_id, joints, origin_y = robots[index]
            target = genome.target_at(step, steps)
            apply_target_pose(robot_id, joints, target)
            energies[index] += float(np.sum(np.abs(target)))

        p.stepSimulation()

        if not p.isConnected():
            break

        for index, (robot_id, _joints, _origin_y) in enumerate(robots):
            head_height_sums[index] += p.getLinkState(robot_id, NECK)[0][2]

        if gui and p.isConnected():
            for index, (robot_id, _joints, origin_y) in enumerate(robots):
                primitive_name = population[index].primitive_name_at(step, steps)
                label = f"g{generation} #{index} {primitive_name}"
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
    if not p.isConnected():
        return (average_head_heights - energies * 0.00002).tolist()

    scores = []
    for index, (robot_id, _joints, origin_y) in enumerate(robots):
        scores.append(
            fitness(
                robot_id,
                origin_y,
                float(average_head_heights[index]),
                float(energies[index]),
            )
        )
    return scores


def tournament(population: list[tuple[float, Genome]], size: int = 3) -> Genome:
    contenders = random.sample(population, k=min(size, len(population)))
    contenders.sort(key=lambda item: item[0], reverse=True)
    return contenders[0][1]


def crossover(a: Genome, b: Genome) -> Genome:
    mask = np.random.random(a.weights.shape) < 0.5
    child_weights = np.where(mask, a.weights, b.weights)

    if random.random() < 0.5:
        first_four = list(a.order[:4])
    else:
        first_four = list(b.order[:4])

    if random.random() < 0.4:
        random.shuffle(first_four)

    child_order = np.array(first_four + [4, 5])
    return Genome(weights=child_weights.copy(), order=child_order)


def mutate(genome: Genome, rate: float, scale: float) -> Genome:
    weights = genome.weights.copy()
    mask = np.random.random(weights.shape) < rate
    weights[mask] += np.random.normal(0.0, scale, size=weights[mask].shape)

    order = genome.order.copy()
    if random.random() < rate:
        swap_a, swap_b = random.sample(range(4), k=2)
        order[swap_a], order[swap_b] = order[swap_b], order[swap_a]

    order[4:] = [4, 5]
    return Genome(weights=np.clip(weights, 0.35, 1.65), order=order)


def next_generation(
    scored: list[tuple[float, Genome]],
    population_size: int,
    elite_count: int,
    mutation_rate: float,
    mutation_scale: float,
) -> list[Genome]:
    scored.sort(key=lambda item: item[0], reverse=True)
    parent_pool = scored[:2]
    elite_count = min(elite_count, 2)
    next_pop = [
        Genome(weights=g.weights.copy(), order=g.order.copy())
        for _, g in scored[:elite_count]
    ]

    while len(next_pop) < population_size:
        parent_a = random.choice(parent_pool)[1]
        parent_b = random.choice(parent_pool)[1]
        child = crossover(parent_a, parent_b)
        next_pop.append(mutate(child, mutation_rate, mutation_scale))

    return next_pop


def train(args: argparse.Namespace) -> Genome:
    population = [Genome.random() for _ in range(args.population)]
    best: tuple[float, Genome] | None = None

    connect(gui=args.watch_population)
    try:
        for generation in range(args.generations):
            scores = evaluate_population(
                population,
                args.steps,
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
                f"order={order_label(scored[0][1])}"
            )

            if args.watch_population and not p.isConnected():
                break

            population = next_generation(
                scored,
                population_size=args.population,
                elite_count=args.elites,
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
        while p.isConnected():
            score = evaluate_population([genome], args.steps, gui=True)[0]
            print(f"replay score={score:.3f}")
            time.sleep(0.8)
    except KeyboardInterrupt:
        pass
    finally:
        safe_disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--mutation-rate", type=float, default=0.12)
    parser.add_argument("--mutation-scale", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--watch-population", action="store_true")
    parser.add_argument("--replay", action="store_true")
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
