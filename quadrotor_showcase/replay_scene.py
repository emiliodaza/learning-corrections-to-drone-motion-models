"""Shared camera and Python drawing for saved Isaac Sim flight poses."""

from math import cos, pi, sin

import numpy as np

from quadrotor_control import BRAKE_SECONDS, HEIGHT, MOTOR_XY, reference, rotation_matrix
from run_isaac import MODEL_NAMES, VIDEO_FPS, load_replay


VIEW_SIZE = (600, 448)
COLORS = ((59, 69, 80), (0, 108, 160), (174, 67, 25))
BACKGROUND = (245, 248, 250)
VIEW_ANGLE = 25 * pi / 180
CAMERA_NORMAL = np.array([0, -sin(VIEW_ANGLE), cos(VIEW_ANGLE)])
UP = np.array([0, cos(VIEW_ANGLE), sin(VIEW_ANGLE)])
MAX_SCALE = 222.0


class ReplayScene:
    """Use one orthographic scale and the same reference origin in every view."""

    def __init__(self, path):
        self.positions, self.orientations = load_replay(path)
        self.times = (np.arange(len(self.positions)) + 1) / VIDEO_FPS
        self.reference_x = np.array([reference(t, 6.0, 0)[0][0] for t in self.times])
        self.lanes = 2.0 * (np.arange(len(MODEL_NAMES)) - 1)
        self.relative = self.positions - self.origins(np.arange(len(self.positions)))

        # choosing a shared scale that fits the largest excursion from the moving reference
        projected_y = (self.relative + np.array([0, 0, HEIGHT / 2])) @ UP
        ground_y = self.relative[:, :, 1] * UP[1] - HEIGHT / 2 * UP[2]
        half_width = np.max(np.abs(self.relative[:, :, 0])) + 0.38
        half_height = max(np.max(np.abs(projected_y)), np.max(np.abs(ground_y))) + 0.22
        self.scale = min(MAX_SCALE, (VIEW_SIZE[0] / 2 - 24) / half_width,
                         (VIEW_SIZE[1] / 2 - 16) / half_height)
        # widening the view during braking and easing into a closer hover view
        fraction = np.clip((self.times - 2.6) / 1.8, 0, 1)
        zoom = .75 + .25 * (fraction**2 * (3 - 2 * fraction))
        self.scales = self.scale * zoom

    def origins(self, frames):
        shape = (*np.shape(frames), len(self.lanes), 3)
        origins = np.empty(shape)
        origins[..., 0] = self.reference_x[frames][..., None]
        origins[..., 1] = self.lanes
        origins[..., 2] = HEIGHT
        return origins

    def project(self, points, frame):
        points = np.asarray(points)
        x = VIEW_SIZE[0] / 2 + self.scales[frame] * points[..., 0]
        y = VIEW_SIZE[1] / 2 - self.scales[frame] * (points @ UP + HEIGHT / 2 * UP[2])
        return np.stack((x, y), axis=-1)

    def playback(self, gusts):
        """Sample quiet spans at 4× and keep active motion at recorded speed."""
        speeds = VIDEO_FPS * np.linalg.norm(np.diff(self.positions, axis=0, prepend=self.positions[:1]), axis=-1)
        errors = np.linalg.norm(self.relative, axis=-1)
        quiet = (np.max(speeds, axis=1) < .08) & (np.max(errors, axis=1) < .10)
        quiet &= self.times >= BRAKE_SECONDS
        for start, end, _ in gusts:
            quiet[(self.times >= start - 0.3) & (self.times <= end + 0.8)] = False

        # speeding up hover intervals of at least 1.5 seconds while retaining their boundaries
        changes = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
        fast = np.zeros(len(self.times), dtype=bool)
        for start, end in changes.reshape(-1, 2):
            if end - start >= round(1.5 * VIDEO_FPS):
                fast[start + 1:end - 1] = True
        index = 0
        while index < len(self.times):
            rate = 4 if index + 4 <= len(self.times) and fast[index:index + 4].all() else 1
            yield index, rate
            index += rate


def box(center, half_size, angle, tone):
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * half_size
    yaw = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    vertices = vertices @ yaw.T + center
    sides = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return [(vertices[list(face)], tone * .62) for face in sides] + [(vertices[4:], tone)]


def airframe_faces():
    """Build a small quadrotor mesh in body coordinates, in meters."""
    faces = []
    for angle in (-pi / 4, pi / 4):
        faces.extend(box((0, 0, 0), (.185, .014, .013), angle, .72))
    faces.extend(box((.015, 0, 0), (.135, .08, .04), 0, 1.0))
    circle = np.column_stack((np.cos(np.arange(24) * pi / 12), np.sin(np.arange(24) * pi / 12)))
    for x, y in MOTOR_XY:
        rim = np.column_stack((x + .075 * circle[:, 0], y + .075 * circle[:, 1], np.full(24, .061)))
        faces.append((rim, .9))
        cap = np.column_stack((x + .056 * circle[:, 0], y + .056 * circle[:, 1], np.full(24, .062)))
        faces.append((cap, .29))
        hub = np.column_stack((x + .017 * circle[:, 0], y + .017 * circle[:, 1], np.full(24, .068)))
        faces.append((hub, 1.0))
    faces.append((np.array([[.085, .040, .042], [.085, -.040, .042], [.136, 0, .042]]), 1.35))
    return faces


AIRFRAME = airframe_faces()


def draw_view(scene, frame, controller, native_mask=None):
    """Draw ground references, then the actual recorded pose on top."""
    from PIL import Image, ImageDraw, ImageFilter

    supersample = 2
    width, height = VIEW_SIZE
    image = Image.new("RGB", (width * supersample, height * supersample), BACKGROUND)
    draw = ImageDraw.Draw(image)
    color = COLORS[controller]
    position = scene.relative[frame, controller]
    ground = position.copy()
    ground[2] = -HEIGHT
    target = (0, 0, -HEIGHT)

    def point(xyz):
        return tuple(scene.project(xyz, frame) * supersample)

    def line(points, fill, width):
        draw.line([point(p) for p in points], fill=fill, width=width * supersample, joint="curve")

    def marker(position, radius, *, fill=None, outline=None, width=1):
        x, y = point(position)
        radius *= supersample
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline,
                     width=width * supersample)

    # anchoring the half-meter grid to the floor as the view follows the reference
    x_reference = scene.reference_x[frame]
    for x_world in np.arange(np.floor((x_reference - 5) * 2) / 2, x_reference + 5, .5):
        x = x_world - x_reference
        line(((x, -4, -HEIGHT), (x, 4, -HEIGHT)), (218, 226, 233), 1)
    for y in np.arange(-4, 4.5, .5):
        line(((-5, y, -HEIGHT), (5, y, -HEIGHT)), (218, 226, 233), 1)
    line(((-5, 0, -HEIGHT), (5, 0, -HEIGHT)), (175, 189, 201), 1)

    # showing the last second of the drone's ground track beside the target ring
    first = max(0, frame - VIDEO_FPS)
    trail = scene.positions[first:frame + 1, controller].copy() - scene.origins(frame)[controller]
    trail[:, 2] = -HEIGHT
    if len(trail) > 1:
        trail_color = tuple(int(.63 * c + .37 * b) for c, b in zip(color, BACKGROUND))
        line(trail, trail_color, 3)
    marker(target, 13, outline=(79, 95, 111), width=2)
    marker(target, 3, fill=(79, 95, 111))
    line((ground, position), (138, 153, 168), 1)
    marker(ground, 5, fill=color, outline=BACKGROUND)

    if native_mask is None:
        # rotating the mesh with the saved pose and drawing faces from back to front
        rotation = rotation_matrix(scene.orientations[frame, controller])
        faces = [(vertices @ rotation.T + position, tone) for vertices, tone in AIRFRAME]
        for vertices, tone in sorted(faces, key=lambda item: float(item[0].mean(axis=0) @ CAMERA_NORMAL)):
            fill = tuple(min(255, round(tone * channel)) for channel in color)
            draw.polygon([point(vertex) for vertex in vertices], fill=fill)

    image = image.resize(VIEW_SIZE, Image.Resampling.LANCZOS)
    if native_mask is not None:
        alpha = native_mask.getchannel("A")
        image.paste((48, 58, 70), (0, 0, *VIEW_SIZE), alpha.filter(ImageFilter.MaxFilter(3)))
        image.paste(native_mask, (0, 0), alpha)
    return image
