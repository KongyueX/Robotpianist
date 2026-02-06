"""
Usage:
python export_actions.py \
  --ckpt-dir /home/xiaoyi/Robotpianist/robopianist-rl/robopianist_runs/rl/SAC-Twinkle-offline-seed42/best/checkpoint_best \
  --out-dir  /home/xiaoyi/Robotpianist/robopianist-rl/robopianist_runs/rl/SAC-Twinkle-offline-seed42/best/export \
  
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
from dataclasses import dataclass
import json
import numpy as np
import tyro
from flax.training import checkpoints

import mujoco  # for mj_id2name fallback

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

import sac
import specs


# ---------------------------
# Wrapper: record the action passed to the *pre-canonical* env.
# CanonicalSpecWrapper canonical->physical 后，tap 会看到 physical action
# ---------------------------
class ActionTapWrapper:
    def __init__(self, env: Any):
        self._env = env
        self.last_action: Optional[np.ndarray] = None

    def reset(self, *args, **kwargs):
        self.last_action = None
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return self._env.step(action)

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def _maybe_get_physics(env) -> Optional[Any]:
    """Try unwrap common wrapper chains to find dm_control physics."""
    seen = set()
    cur = env
    for _ in range(80):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        phys = getattr(cur, "physics", None)
        if phys is not None:
            return phys

        for attr in ("environment", "_environment", "env", "_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def _extract_actuator_meta(env) -> dict:
    """Extract actuator names, joint names and ctrlrange from compiled mujoco model."""
    meta: Dict[str, Any] = {}
    phys = _maybe_get_physics(env)
    if phys is None:
        return meta

    m = phys.model
    nu = int(m.nu)
    meta["nu"] = nu
    meta["actuator_ctrlrange"] = np.asarray(m.actuator_ctrlrange).tolist()

    act_names: List[str] = []
    joint_names: List[str] = []

    trnid = np.asarray(m.actuator_trnid)[:, 0]  # actuator -> joint id

    for i in range(nu):
        # actuator name
        try:
            act_names.append(m.actuator(i).name)
        except Exception:
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            act_names.append("" if n is None else str(n))

        # joint name
        jid = int(trnid[i])
        try:
            joint_names.append(m.joint(jid).name)
        except Exception:
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            joint_names.append("" if n is None else str(n))

    meta["actuator_names"] = act_names
    meta["actuator_joint_names"] = joint_names
    return meta


@dataclass(frozen=True)
class ExportArgs:
    # ==== checkpoint ====
    ckpt_dir: str
    out_dir: str = "export_out"
    eval_seed: int = 42

    # ==== env (尽量和训练一致) ====
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"

    n_steps_lookahead: int = 10
    trim_silence: bool = True
    gravity_compensation: bool = True
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False

    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = True

    # wrappers
    frame_stack: int = 1
    clip: bool = True
    action_reward_observation: bool = True  # 你这次训练开了

    # recording（只要 out_dir/rollout 下产物即可）
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"

    # agent（默认按你 run.sh；如果不一致可用 CLI 覆盖）
    discount: float = 0.8
    agent_config: sac.SACConfig = sac.SACConfig(
        critic_dropout_rate=0.01,
        critic_layer_norm=True,
        hidden_dims=(256, 256, 256),
    )


def get_env(args: ExportArgs, record_dir: Optional[Path] = None):
    """
    Returns:
      env: wrapped env (canonical action space)
      tap: ActionTapWrapper sitting below CanonicalSpecWrapper, records physical actions
    """
    if record_dir is not None:
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)

    env = suite.load(
        environment_name=args.environment_name,
        seed=args.eval_seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )

    # === 关键：录制 wrapper 必须在 canonical 之前，且 record_dir 指向 out_dir/rollout ===
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=args.record_every)
        env = robopianist_wrappers.MidiEvaluationWrapper(environment=env, deque_size=args.record_every)
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)

    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)

    env = wrappers.ConcatObservationWrapper(env)

    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(env, num_frames=args.frame_stack, flatten=True)

    # --- Insert tap BELOW CanonicalSpecWrapper ---
    tap = ActionTapWrapper(env)
    env = wrappers.CanonicalSpecWrapper(tap, clip=args.clip)

    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env, tap


def main(args: ExportArgs):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 必须叫 rollout（与你 onehand 一致）
    record_dir = out_dir / "rollout"
    record_dir.mkdir(parents=True, exist_ok=True)

    env, tap = get_env(args, record_dir=record_dir)

    # 固定随机性（尽量）
    try:
        env.random_state.seed(args.eval_seed)
    except Exception:
        pass

    # spec + agent init + restore
    spec = specs.EnvironmentSpec.make(env)
    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.eval_seed,
        discount=args.discount,
    )
    agent = checkpoints.restore_checkpoint(
        ckpt_dir=args.ckpt_dir,
        target=agent,
    )

    # rollout（deterministic）
    ts = env.reset()

    actions_canonical: List[np.ndarray] = []
    actions_ctrl: List[np.ndarray] = []

    rewards: List[float] = []
    discounts: List[float] = []
    dones: List[bool] = []

    while True:
        if ts.last():
            break

        a = agent.eval_actions(ts.observation)  # canonical
        a = np.asarray(a)
        actions_canonical.append(a)

        ts = env.step(a)

        if tap.last_action is None:
            raise RuntimeError(
                "ActionTapWrapper did not capture any action. "
                "这意味着 wrapper 顺序不对，或者 CanonicalSpecWrapper 没有把动作传下去。"
            )
        actions_ctrl.append(tap.last_action.copy())

        rewards.append(float(ts.reward))
        discounts.append(float(ts.discount))
        dones.append(bool(ts.last()))

    actions_canonical_arr = np.asarray(actions_canonical)
    actions_ctrl_arr = np.asarray(actions_ctrl)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    discounts_arr = np.asarray(discounts, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.bool_)

    # ---- Save files（命名严格按你 onehand） ----
    np.save(out_dir / "actions_canonical.npy", actions_canonical_arr)
    np.save(out_dir / "actions_ctrl.npy", actions_ctrl_arr)
    np.save(out_dir / "actions.npy", actions_ctrl_arr)  # 默认部署用 physical

    np.savez(
        out_dir / "traj.npz",
        actions_canonical=actions_canonical_arr,
        actions_ctrl=actions_ctrl_arr,
        rewards=rewards_arr,
        discounts=discounts_arr,
        dones=dones_arr,
        eval_seed=np.int32(args.eval_seed),
    )

    # ---- Meta（字段对齐 onehand 习惯） ----
    meta: Dict[str, Any] = {
        "ckpt_dir": str(args.ckpt_dir),
        "out_dir": str(out_dir),
        "eval_seed": int(args.eval_seed),
        "environment_name": args.environment_name,
        "control_timestep": float(args.control_timestep),

        "action_dim_expected": int(spec.action_dim),

        "actions_canonical_shape": list(actions_canonical_arr.shape),
        "actions_ctrl_shape": list(actions_ctrl_arr.shape),

        "canonical_action_range_expected": [-1.0, 1.0],
        "actions_canonical_minmax": [
            float(actions_canonical_arr.min()) if actions_canonical_arr.size else 0.0,
            float(actions_canonical_arr.max()) if actions_canonical_arr.size else 0.0,
        ],
        "actions_ctrl_minmax": [
            float(actions_ctrl_arr.min()) if actions_ctrl_arr.size else 0.0,
            float(actions_ctrl_arr.max()) if actions_ctrl_arr.size else 0.0,
        ],

        "actions_npy_is": "actions_ctrl",

        "record_dir": str(record_dir),
        "record_every": int(args.record_every),
        "camera_id": args.camera_id,
        "record_resolution": list(args.record_resolution),

        "task_kwargs": {
            "n_steps_lookahead": args.n_steps_lookahead,
            "trim_silence": args.trim_silence,
            "gravity_compensation": args.gravity_compensation,
            "reduced_action_space": args.reduced_action_space,
            "control_timestep": args.control_timestep,
            "wrong_press_termination": args.wrong_press_termination,
            "disable_fingering_reward": args.disable_fingering_reward,
            "disable_forearm_reward": args.disable_forearm_reward,
            "disable_colorization": args.disable_colorization,
            "disable_hand_collisions": args.disable_hand_collisions,
            "primitive_fingertip_collisions": args.primitive_fingertip_collisions,
            "stretch_factor": args.stretch_factor,
            "shift_factor": args.shift_factor,
        },
    }

    meta.update(_extract_actuator_meta(env))

    if hasattr(env, "get_statistics"):
        try:
            meta["stats"] = env.get_statistics()
        except Exception:
            pass
    if hasattr(env, "get_musical_metrics"):
        try:
            meta["music"] = env.get_musical_metrics()
        except Exception:
            pass

    # 尽量拿到 latest_filename（取不到也不影响文件实际写入 rollout 目录）
    latest = getattr(env, "latest_filename", None)
    if latest is not None:
        meta["latest_video"] = str(latest)

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(tyro.cli(ExportArgs))
