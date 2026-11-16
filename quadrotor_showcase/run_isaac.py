"""Run the quadrotor braking benchmark inside NVIDIA Isaac Sim 6.2."""

import argparse
import csv
from pathlib import Path

import numpy as np

from quadrotor_control import (
    BRAKING_GUSTS, DT, FLIGHT_SECONDS, GRAVITY, HEIGHT, HOVER_GUST, INERTIA,
    MASS, MOTOR_XY, SHOWCASE_GUSTS, SHOWCASE_SECONDS,
    learned_acceleration, load_checkpoint, motor_command, plant_wrench,
    reference, rotation_matrix, score_flight,
)


MODEL_NAMES = ("thrust", "direct", "residual")
COLORS = ((0.72, 0.74, 0.79), (0.16, 0.54, 0.95), (0.98, 0.38, 0.12))
VIDEO_FPS = 30


def create_scene(models, physics=True):
    """Build matching quadrotor geometry for simulation and replay."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if physics:
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(GRAVITY)

    def box(path, position, size, color, yaw=0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        if yaw:
            cube.AddRotateZOp().Set(yaw)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return cube

    def disc(path, position, radius, height, color):
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(*position))
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return cylinder

    # placing each controller in its own lane with identical geometry and dynamics
    paths = []
    colliders = []
    lanes = (np.arange(len(models)) - (len(models) - 1) / 2) * 2.0
    box("/World/Floor", (4.5, 0, -0.06), (15, 9, 0.12), (0.18, 0.22, 0.28))
    for index, (model, lane) in enumerate(zip(models, lanes)):
        path = f"/World/{model}_quadrotor"
        paths.append(path)
        color = COLORS[index]
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        hull = box(f"{path}/Hull", (0.015, 0, 0), (0.27, 0.16, 0.08), color)
        if physics:
            UsdPhysics.RigidBodyAPI.Apply(body)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateMassAttr(MASS)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, INERTIA)))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0)))
            UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
            colliders.append(hull.GetPath())
        for arm, yaw in enumerate((-45, 45), start=1):
            box(f"{path}/Arm_{arm}", (0, 0, 0), (0.36, 0.025, 0.025), color, yaw)
        for number, (x, y) in enumerate(MOTOR_XY, start=1):
            disc(f"{path}/Rotor_{number}", (float(x), float(y), 0.055), 0.072, 0.022, color)

        box(f"/World/Lane_{index}", (4.5, float(lane), 0.005), (11, 0.033, 0.01), color)
        disc(f"/World/Goal_{index}", (9.0, float(lane), 0.012), 0.27, 0.018, color)

    # preventing contact between drones so one controller cannot affect another trial
    if physics:
        group = UsdPhysics.CollisionGroup.Define(stage, "/World/Quadrotors")
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(colliders)
        group.CreateFilteredGroupsRel().AddTarget(group.GetPath())

    light = UsdLux.DomeLight.Define(stage, "/World/Fill")
    light.CreateIntensityAttr(600.0)
    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(1600.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -15))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(24.0)
    view = Gf.Matrix4d(1).SetLookAt(
        Gf.Vec3d(4.5, -12.5, 8.8), Gf.Vec3d(4.5, 0, 1.1), Gf.Vec3d(0, 0, 1)
    )
    camera.AddTransformOp().Set(view.GetInverse())
    return paths, lanes, str(camera.GetPath())


def simulate_trial(body, models, lanes, initial_speed, seed, checkpoint, smoke,
                   duration=FLIGHT_SECONDS, gusts=BRAKING_GUSTS):
    """Read the PhysX state and apply the motor and drag wrench at 120 Hz."""
    from isaacsim.core.simulation_manager import SimulationManager

    coefficient, mean, scale, weights = checkpoint
    duration = 1.5 if smoke else duration
    times = np.arange(round(duration / DT) + 1) * DT
    positions = np.empty((len(times), len(models), 3))
    velocities = np.empty_like(positions)
    targets = np.empty_like(positions)
    orientations = np.empty((len(times), len(models), 4))

    # resetting every drone to the same starting state apart from its lane
    start_positions = np.column_stack((np.zeros(len(models)), lanes, np.full(len(models), HEIGHT)))
    start_velocities = np.zeros_like(start_positions)
    start_velocities[:, 0] = 0.6 if smoke else initial_speed
    body.set_world_poses(
        positions=start_positions.astype(np.float32),
        orientations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(models), 1)),
    )
    body.set_velocities(
        linear_velocities=start_velocities.astype(np.float32),
        angular_velocities=np.zeros_like(start_velocities, dtype=np.float32),
    )
    motor_speeds = np.full((len(models), 4), np.sqrt(GRAVITY / (4 * coefficient)))
    motor_response = 1 - np.exp(-DT / 0.04)
    forces = np.empty((len(models), 3), dtype=np.float32)
    torques = np.empty_like(forces)
    for step, time in enumerate(times):
        position, orientation = body.get_world_poses()
        velocity, angular_velocity = body.get_velocities()
        positions[step] = position.numpy()
        velocities[step] = velocity.numpy()
        orientations[step] = orientation.numpy()
        angular_velocities = angular_velocity.numpy()
        if not (np.isfinite(positions[step]).all() and np.isfinite(velocities[step]).all()
                and np.isfinite(orientations[step]).all() and np.isfinite(angular_velocities).all()):
            raise RuntimeError(f"PhysX returned a non-finite state at {time:.3f} seconds")

        for index, model in enumerate(models):
            position = positions[step, index]
            velocity = velocities[step, index]
            rotation = rotation_matrix(orientations[step, index])
            rates = rotation.T @ angular_velocities[index]
            target = reference(time, initial_speed, lanes[index])
            targets[step, index] = target[0]

            # using only body x and y predictions, where the nominal thrust term is zero
            lateral = np.zeros(2)
            if model != "thrust":
                acceleration = learned_acceleration(
                    rotation, velocity, rates, motor_speeds[index], mean, scale, weights[model, seed]
                )
                # rotating the prediction into world coordinates and tapering it near hover
                lateral = (rotation[:, :2] @ acceleration[:2])[:2]
                lateral *= 3.0 / max(3.0, np.linalg.norm(lateral))
                lateral *= min(1.0, np.linalg.norm(velocity[:2]) / 2.0)

            commanded = motor_command(position, velocity, rotation, rates, target, lateral, coefficient)
            # modeling rotor lag before computing the forces applied to physx
            motor_speeds[index] += motor_response * (commanded - motor_speeds[index])
            forces[index], torques[index] = plant_wrench(
                time, velocity, rotation, motor_speeds[index], coefficient, gusts
            )

        if step < len(times) - 1:
            body.apply_forces_and_torques_at_pos(forces=forces, torques=torques, local_frame=False)
            SimulationManager.step()

    return times, positions, velocities, targets, orientations


def render_poses(models, positions, orientations, output):
    """Render saved PhysX poses in a visual-only stage at 30 fps."""
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom

    paths, _, camera = create_scene(models, physics=False)
    stage = omni.usd.get_context().get_stage()
    transforms = []
    for path in paths:
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        transforms.append((xform.AddTranslateOp(), xform.AddOrientOp()))

    frame_dir = output / "frames"
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(camera, (1280, 720))
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(frame_dir))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True)
    writer.attach(product)

    # reusing one camera output while updating each body from the saved poses
    for frame, (positions_at_frame, orientations_at_frame) in enumerate(
        zip(positions, orientations), start=1
    ):
        for (translate, orient), position, quaternion in zip(
            transforms, positions_at_frame, orientations_at_frame
        ):
            translate.Set(Gf.Vec3d(*map(float, position)))
            orient.Set(Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*map(float, quaternion[1:]))))
        rep.orchestrator.step(delta_time=1 / VIDEO_FPS, rt_subframes=1)
        if frame % VIDEO_FPS == 0:
            print(f"Rendered {frame}/{len(positions)} frames", flush=True)

    rep.orchestrator.wait_until_complete()
    writer.detach()
    product.destroy()
    frames = sum(1 for _ in frame_dir.rglob("rgb_*.png"))
    if frames != len(positions):
        raise RuntimeError(f"Expected {len(positions)} RGB frames in {frame_dir}, found {frames}")


def load_replay(path):
    with np.load(path, allow_pickle=False) as saved:
        positions = saved["positions"]
        orientations = saved["orientations"]
    if (positions.ndim != 3 or positions.shape[1:] != (len(MODEL_NAMES), 3)
            or len(positions) == 0
            or orientations.shape != (len(positions), len(MODEL_NAMES), 4)
            or not np.isfinite(positions).all() or not np.isfinite(orientations).all()):
        raise ValueError(f"Invalid 6 m/s replay data in {path}")
    return positions, orientations


def run(args, checkpoint, output):
    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": True,
        "disable_viewport_updates": True,
        "multi_gpu": False,
        "active_cuda_gpus": [0],
    })
    try:
        if args.replay is not None:
            render_poses(MODEL_NAMES, *load_replay(args.replay), output)
            print(f"Recording finished. Output in {output}", flush=True)
            return

        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager

        models = ("thrust",) if args.smoke else MODEL_NAMES
        paths, lanes, _ = create_scene(models)
        SimulationManager.setup_simulation(dt=DT, device="cpu")
        SimulationManager.initialize_physics()
        body = RigidPrim(paths, resolve_paths=False)

        if args.smoke:
            _, positions, _, _, orientations = simulate_trial(body, models, lanes, 0.0, 42, checkpoint, True)
            if np.linalg.norm(positions[-1, 0] - positions[0, 0]) < 0.02 or np.any(positions[:, 0, 2] < 1.7):
                raise RuntimeError("Smoke test did not produce the expected controlled hover")
            SimulationManager.invalidate_physics()
            render_poses(models, positions[-1:], orientations[-1:], output)
            print(f"Smoke test passed. Output in {output}", flush=True)
            return

        duration = SHOWCASE_SECONDS if args.long_run else FLIGHT_SECONDS
        gusts = SHOWCASE_GUSTS if args.long_run else BRAKING_GUSTS
        hover_gust = HOVER_GUST if args.long_run else None
        print(f"Simulating {duration:g} seconds for each trial", flush=True)
        metrics = []
        frameskip = round(1 / (DT * VIDEO_FPS))
        with (output / "trajectories.csv").open("w", newline="") as stream:
            trajectory = csv.writer(stream)
            trajectory.writerow([
                "speed_mps", "seed", "model", "time_s", "x", "y", "z", "vx", "vy", "vz",
                "reference_x", "reference_y", "reference_z",
            ])
            for speed in (3.0, 6.0):
                for seed in (42, 43, 44):
                    time, position, velocity, target, orientation = simulate_trial(
                        body, models, lanes, speed, seed, checkpoint, False, duration, gusts
                    )
                    # saving the showcase trial at 30 fps while keeping full-rate data in the csv
                    if speed == 6.0 and seed == 42:
                        np.savez_compressed(
                            output / "replay.npz",
                            positions=position[frameskip::frameskip],
                            orientations=orientation[frameskip::frameskip],
                            gusts=np.asarray(gusts),
                        )
                    for index, model in enumerate(models):
                        p, v, goal = position[:, index], velocity[:, index], target[:, index]
                        scores = score_flight(time, p, v, goal, hover_gust)
                        metrics.append({"speed_mps": speed, "seed": seed, "model": model, **scores})
                        trajectory.writerows(
                            [speed, seed, model, t, *xyz, *vel, *ref]
                            for t, xyz, vel, ref in zip(time, p, v, goal)
                        )
                    print(f"Finished {speed:g} m/s, seed {seed}", flush=True)
        with (output / "flight_metrics.csv").open("w", newline="") as stream:
            metric_writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            metric_writer.writeheader()
            metric_writer.writerows(metrics)
        print(f"Benchmark finished. Output in {output}", flush=True)

        if args.record:
            SimulationManager.invalidate_physics()
            render_poses(models, *load_replay(output / "replay.npz"), output)
            print(f"Recording finished. Output in {output}", flush=True)
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="NeuroBEM .npz exported from Colab")
    parser.add_argument("--output", type=Path, help="Results directory")
    parser.add_argument("--record", action="store_true", help="Render the 6 m/s trial for seed 42")
    parser.add_argument("--replay", type=Path, help="Render a saved replay.npz without running physics")
    parser.add_argument("--smoke", action="store_true", help="Test physics and RGB capture without weights")
    parser.add_argument("--long-run", action="store_true", help="Simulate 24 seconds with a second gust during hover")
    args = parser.parse_args()
    if args.long_run and (args.smoke or args.replay is not None):
        parser.error("--long-run requires a checkpoint and a new physics run")
    if args.replay is not None:
        if args.checkpoint is not None or args.record or args.smoke:
            parser.error("--replay runs on its own")
        checkpoint = None
    elif args.smoke:
        checkpoint = GRAVITY / (4 * 650**2), None, None, {}
    elif args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint)
    else:
        parser.error("--checkpoint is required unless you use --smoke or --replay")
    default_output = "results/isaac_smoke" if args.smoke else "results/isaac_braking"
    if args.replay is not None:
        default_output = args.replay.parent
    output = (args.output or Path(default_output)).resolve()
    frame_dir = output / "frames"
    if (args.record or args.smoke or args.replay is not None) and frame_dir.exists() and any(frame_dir.iterdir()):
        parser.error(f"Use a fresh --output directory for recording. {frame_dir} already contains files.")
    output.mkdir(parents=True, exist_ok=True)
    run(args, checkpoint, output)


if __name__ == "__main__":
    main()
