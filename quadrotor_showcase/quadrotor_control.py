"""NeuroBEM inference, quadrotor control, and flight metrics."""

import numpy as np


DT = 1 / 120
BRAKE_SECONDS = 3.0
FLIGHT_SECONDS = 9.0
SHOWCASE_SECONDS = 24.0
BRAKING_GUSTS = ((1.0, 2.0, 2.0),)
HOVER_GUST = (10.0, 12.0, -3.0)
SHOWCASE_GUSTS = (*BRAKING_GUSTS, HOVER_GUST)
HEIGHT = 2.3
GRAVITY = 9.81
MASS = 0.772
INERTIA = np.array([0.0025, 0.0025, 0.0043])

# following the neurobem motor order in a forward, left, and up body frame
MOTOR_XY = 0.12 * np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
MOTOR_MIX = np.array([
    np.ones(4), MOTOR_XY[:, 1], -MOTOR_XY[:, 0], [0.016, -0.016, -0.016, 0.016]
])
INVERSE_MIX = np.linalg.inv(MOTOR_MIX)
INPUT_COLUMNS = [
    "vel x", "vel y", "vel z", "ang vel x", "ang vel y", "ang vel z",
    "mot 1", "mot 2", "mot 3", "mot 4",
]


def load_checkpoint(path):
    """Load the training-fitted scaler, thrust coefficient, and both NNs."""
    with np.load(path, allow_pickle=False) as checkpoint:
        if checkpoint["input_columns"].tolist() != INPUT_COLUMNS:
            raise ValueError("Unexpected feature order in the checkpoint")
        mean = checkpoint["scaler_mean"]
        scale = checkpoint["scaler_scale"]
        coefficient = float(checkpoint["thrust_coefficient"].item())
        weights = {}
        for seed in (42, 43, 44):
            for model in ("direct", "residual"):
                weights[model, seed] = tuple(
                    checkpoint[f"{model}_{seed}_{key}"]
                    for key in ("0.weight", "0.bias", "2.weight", "2.bias")
                )

    if (mean.shape != (10,) or scale.shape != (10,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(coefficient) or coefficient <= 0):
        raise ValueError("Invalid feature scaler or thrust coefficient")
    for key, layers in weights.items():
        if tuple(layer.shape for layer in layers) != ((32, 10), (32,), (3, 32), (3,)):
            raise ValueError(f"Unexpected layer shapes for {key}")
        if any(not np.isfinite(layer).all() for layer in layers):
            raise ValueError(f"Non-finite weights for {key}")
    return coefficient, mean, scale, weights


def rotation_matrix(quaternion):
    """Convert an Isaac Sim (w, x, y, z) orientation to body-to-world rotation."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def reference(time, initial_speed, lane):
    """Follow a half-cosine braking profile, then hold the final waypoint."""
    phase = np.pi * min(time, BRAKE_SECONDS) / BRAKE_SECONDS
    distance = initial_speed * BRAKE_SECONDS / (2 * np.pi) * (phase + np.sin(phase))
    speed = initial_speed / 2 * (1 + np.cos(phase))
    acceleration = -initial_speed * np.pi / (2 * BRAKE_SECONDS) * np.sin(phase)
    return (
        np.array([distance, lane, HEIGHT]),
        np.array([speed, 0, 0]),
        np.array([acceleration, 0, 0]),
    )


def learned_acceleration(rotation, velocity, rates, motor_speeds, mean, scale, weights):
    # matching training features with body-frame velocities and squared rotor speeds
    features = np.concatenate((rotation.T @ velocity, rates, motor_speeds**2))
    hidden_weight, hidden_bias, output_weight, output_bias = weights
    normalized = ((features - mean) / scale).astype(hidden_weight.dtype, copy=False)
    return output_weight @ np.maximum(hidden_weight @ normalized + hidden_bias, 0) + output_bias


def motor_command(position, velocity, rotation, rates, target, lateral, coefficient):
    """Use a shared position and attitude controller to command four rotor speeds."""
    target_position, target_velocity, target_acceleration = target
    # combining the reference acceleration with feedback before subtracting lateral compensation
    acceleration = (target_acceleration + np.array([1.9, 1.9, 4.0]) * (target_position - position)
                    + np.array([2.9, 2.9, 3.4]) * (target_velocity - velocity))
    acceleration[:2] -= lateral
    acceleration[:2] *= 5.0 / max(5.0, np.linalg.norm(acceleration[:2]))
    acceleration[2] = np.clip(acceleration[2], -4.0, 4.0)
    desired_force = MASS * (acceleration + np.array([0, 0, GRAVITY]))

    # aligning the body z axis with the desired force while holding yaw near zero
    up = desired_force / np.linalg.norm(desired_force)
    left = np.cross(up, [1, 0, 0])
    left /= np.linalg.norm(left)
    desired_rotation = np.column_stack((np.cross(left, up), left, up))
    skew = 0.5 * (desired_rotation.T @ rotation - rotation.T @ desired_rotation)
    attitude_error = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    torque = (-np.array([0.35, 0.35, 0.22]) * attitude_error
              - np.array([0.045, 0.045, 0.038]) * rates
              + np.cross(rates, INERTIA * rates))
    # converting the requested thrust and torque to feasible rotor speeds
    thrust = max(0.0, desired_force @ rotation[:, 2])
    rotor_forces = INVERSE_MIX @ np.r_[thrust, torque]
    max_rotor_force = (1.65**2) * MASS * GRAVITY / 4
    return np.sqrt(np.clip(rotor_forces, 0, max_rotor_force) / (MASS * coefficient))


def plant_wrench(time, velocity, rotation, motor_speeds, coefficient, gusts=BRAKING_GUSTS):
    """Build the shared plant's world-frame force and torque for PhysX."""
    # summing rotor thrust along body z and moments from each arm and spin direction
    thrust, *torque = MOTOR_MIX @ (MASS * coefficient * motor_speeds**2)

    # computing drag from air-relative velocity using the gusts active at this time
    lateral_wind = sum(speed for start, end, speed in gusts if start <= time < end)
    wind = np.array([0, lateral_wind, 0])
    relative_velocity = velocity - wind
    drag = -MASS * (0.20 + 0.04 * np.linalg.norm(relative_velocity)) * relative_velocity
    return thrust * rotation[:, 2] + drag, rotation @ torque


def score_flight(times, positions, velocities, targets, hover_gust=None):
    """Measure tracking, overshoot, and sustained entry into a stable hover."""
    distance = np.linalg.norm(positions - targets, axis=1)
    after_braking = times >= BRAKE_SECONDS
    near_hover = (distance <= 0.25) & (np.linalg.norm(velocities, axis=1) <= 0.25)
    # scanning backward to find when the drone first stays within both limits
    stayed_near_hover = np.logical_and.accumulate(near_hover[::-1])[::-1]
    eligible = after_braking & stayed_near_hover & (times <= times[-1] - 1.0)
    crossed_floor = bool(np.any(positions[:, 2] < 0.15))
    settled = bool(eligible.any()) and not crossed_floor
    scores = {
        "tracking_rmse_m": np.sqrt(np.mean(distance**2)),
        "hover_rmse_m": np.sqrt(np.mean(distance[after_braking]**2)),
        "overshoot_m": max(0.0, np.max(positions[:, 0] - targets[-1, 0])),
        "settling_s_after_brake": times[eligible][0] - BRAKE_SECONDS if settled else None,
        "settled": settled,
        "crossed_floor": crossed_floor,
    }
    if hover_gust is not None:
        start, end, _ = hover_gust
        after_gust = times >= end
        recovered = eligible & after_gust
        lateral_error = np.abs(positions[:, 1] - targets[:, 1])
        scores.update({
            "hover_gust_peak_lateral_m": np.max(lateral_error[times >= start]),
            "post_gust_rmse_m": np.sqrt(np.mean(distance[after_gust]**2)),
            "recovery_s_after_hover_gust": (
                times[recovered][0] - end if recovered.any() and not crossed_floor else None
            ),
        })
    return scores
