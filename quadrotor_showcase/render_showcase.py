"""Capture clean quadrotor silhouettes from the Isaac Sim camera."""

import argparse
from pathlib import Path

import numpy as np

from quadrotor_control import HEIGHT, MOTOR_XY
from replay_scene import CAMERA_NORMAL, COLORS, VIEW_SIZE, ReplayScene
from run_isaac import MODEL_NAMES, VIDEO_FPS, create_scene


RESOLUTION = (len(MODEL_NAMES) * VIEW_SIZE[0], VIEW_SIZE[1])


def palette_image(result, palette):
    """Map Replicator's semantic IDs to exact colors and transparent background."""
    ids = np.asarray(result["data"])
    if ids.shape != (RESOLUTION[1], RESOLUTION[0]):
        raise RuntimeError(f"Expected an Isaac Sim mask of size {RESOLUTION}, got {ids.shape[::-1]}")
    pixels = np.zeros((*ids.shape, 4), dtype=np.uint8)
    visible = set()
    for index, semantics in result["info"]["idToLabels"].items():
        label = semantics.get("class", "") if isinstance(semantics, dict) else semantics
        if label in palette:
            color, controller = palette[label]
            mask = ids == int(index)
            if np.any(mask):
                pixels[mask] = (*color, 255)
                visible.add(controller)
    if visible != set(MODEL_NAMES):
        missing = ", ".join(sorted(set(MODEL_NAMES) - visible))
        raise RuntimeError(f"Isaac Sim's camera mask is missing {missing}")
    return pixels


def render(app, replay, output):
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdSemantics, Vt

    paths, _, camera_path = create_scene(MODEL_NAMES, physics=False)
    app.reset_render_settings()
    stage = omni.usd.get_context().get_stage()
    for path in ("/World/Floor", "/World/Fill", "/World/Sun"):
        stage.RemovePrim(path)

    # framing three equal views in one orthographic camera, each centered on its reference
    center = Gf.Vec3d(0, 0, HEIGHT / 2)
    eye = center + Gf.Vec3d(*map(float, 13 * CAMERA_NORMAL))
    camera = UsdGeom.Camera.Get(stage, camera_path)
    camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    camera.GetOrderedXformOps()[0].Set(
        Gf.Matrix4d(1).SetLookAt(eye, center, Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    horizontal = camera.CreateHorizontalApertureAttr()
    vertical = camera.CreateVerticalApertureAttr()

    palette = {}
    transforms = []

    def label(prim, controller, part, color, tone):
        name = f"{controller}_{part}"
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set(Vt.TokenArray([name]))
        palette[name] = (tuple(round(tone * channel) for channel in color), controller)

    # labeling each airframe part so the segmentation buffer can identify and color it
    for i, (controller, path, color) in enumerate(zip(MODEL_NAMES, paths, COLORS)):
        stage.RemovePrim(f"/World/Lane_{i}")
        stage.RemovePrim(f"/World/Goal_{i}")
        label(stage.GetPrimAtPath(f"{path}/Hull"), controller, "hull", color, 1.0)
        for arm in range(1, 3):
            label(stage.GetPrimAtPath(f"{path}/Arm_{arm}"), controller, "arm", color, .72)
        for rotor, (x, y) in enumerate(MOTOR_XY, start=1):
            label(stage.GetPrimAtPath(f"{path}/Rotor_{rotor}"), controller, "rotor", color, .9)
            cap = UsdGeom.Cylinder.Define(stage, f"{path}/Cap_{rotor}")
            cap.CreateAxisAttr(UsdGeom.Tokens.z)
            cap.CreateRadiusAttr(.053)
            cap.CreateHeightAttr(.004)
            cap.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), .068))
            label(cap.GetPrim(), controller, "cap", color, .29)
        body = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        transforms.append((body.AddTranslateOp(), body.AddOrientOp()))

    masks = output / "masks"
    masks.mkdir()
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(camera_path, RESOLUTION)
    segmentation = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    segmentation.attach(product)
    print(f"Capturing {len(replay.times)} frames of quadrotor masks", flush=True)
    try:
        for frame, (positions, quaternions, scale) in enumerate(zip(replay.relative, replay.orientations, replay.scales)):
            # converting the replay scale to apertures in tenths of a scene unit
            horizontal.Set(float(10 * RESOLUTION[0] / scale))
            vertical.Set(float(10 * RESOLUTION[1] / scale))
            for i, ((translate, orient), position, quaternion) in enumerate(zip(transforms, positions, quaternions)):
                x, y, z = map(float, position)
                translate.Set(Gf.Vec3d(x + (i - 1) * VIEW_SIZE[0] / scale, y, z + HEIGHT))
                orient.Set(Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*map(float, quaternion[1:]))))
            rep.orchestrator.step(delta_time=1 / VIDEO_FPS, rt_subframes=4 if frame == 0 else 1)
            pixels = palette_image(segmentation.get_data(), palette)
            Image.fromarray(pixels).save(masks / f"mask_{frame:04d}.png", compress_level=2)
            if (frame + 1) % VIDEO_FPS == 0:
                print(f"Captured {frame + 1}/{len(replay.times)} frames", flush=True)
    finally:
        segmentation.detach()
        product.destroy()
    print(f"Isaac Sim camera capture finished. Output in {masks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Choose an empty output directory. Found existing files in {args.output}")
    replay = ReplayScene(args.replay)
    args.output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    # using the rtx id buffer for silhouettes and applying fixed colors in python
    app = SimulationApp({
        "headless": True, "disable_viewport_updates": True, "multi_gpu": False,
        "active_gpu": 0, "limit_cpu_threads": 8,
        "renderer": "MinimalRendering", "minimal_shading_mode": 4, "anti_aliasing": 0,
    })
    try:
        render(app, replay, args.output)
    finally:
        app.close()


if __name__ == "__main__":
    main()
