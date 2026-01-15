import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

from robopianist import suite


def get_physics(env):
    # wrapper 容错
    physics = getattr(env, "_physics", None)
    if physics is not None:
        return physics
    for attr in ["_env", "_environment"]:
        obj = getattr(env, attr, None)
        if obj is not None and hasattr(obj, "_physics"):
            return obj._physics
    raise RuntimeError("Cannot find physics handle on env. Inspect env attrs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera_id", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("rollout.mp4"))
    ap.add_argument("--policy", choices=["zero", "random"], default="zero")
    args = ap.parse_args()

    env = suite.load(args.env)
    ts = env.reset()
    spec = env.action_spec()
    physics = get_physics(env)

    writer = imageio.get_writer(str(args.out), fps=args.fps)
    try:
        for _ in range(args.steps):
            if args.policy == "zero":
                action = np.zeros(spec.shape, dtype=spec.dtype)
            else:
                action = np.random.uniform(spec.minimum, spec.maximum).astype(spec.dtype)

            ts = env.step(action)
            frame = physics.render(height=args.height, width=args.width, camera_id=args.camera_id)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"[OK] Saved video: {args.out.resolve()}")


if __name__ == "__main__":
    main()
