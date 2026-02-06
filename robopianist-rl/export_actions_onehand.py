"""
Usage:
python export_actions_onehand.py \
  --ckpt-dir /home/xiaoyi/Robotpianist/robopianist-rl/robopianist_runs/rl/SAC-Twinkle-offline-seed42/best/checkpoint_best \
  --out-dir /home/xiaoyi/Robotpianist/robopianist-rl/robopianist_runs/rl/SAC-Twinkle-offline-seed42/best/export \

"""

# export_actions.py
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass
import json
import numpy as np
import tyro
from flax.training import checkpoints

import mujoco  # for mj_id2name fallback

import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

from mujoco_utils import composer_utils
from robopianist.music import library
from robopianist.models.hands import HandSide
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand

import sac
import specs


# ---------------------------
# Wrapper: record the action that gets passed to the *pre-canonical* env.
# If CanonicalSpecWrapper de-normalizes canonical action -> physical action,
# then this wrapper will see the physical action.
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


@dataclass(frozen=True)
class ExportArgs:
    # ==== checkpoint ====
    ckpt_dir: str  # e.g. robopianist-rl/.../best/checkpoint_best
    out_dir: str = "export_out"
    # 固定 reset 随机性，确保动作序列稳定可复现
    eval_seed: int = 42

    # ==== env (必须和训练一致) ====
    midi_name: str = "TwinkleTwinkleLittleStar"
    hand_side: str = "RIGHT"

    n_steps_lookahead: int = 10
    trim_silence: bool = False
    initial_buffer_time: float = 0.5
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    wrong_press_termination: bool = False

    disable_fingering_reward: bool = False
    disable_colorization: bool = False
    primitive_fingertip_collisions: bool = False

    # wrappers
    frame_stack: int = 1
    clip: bool = True
    action_reward_observation: bool = False

    # recording
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"

    # agent
    agent_config: sac.SACConfig = sac.SACConfig()


def _parse_hand_side(s: str) -> HandSide:
    s = s.upper()
    if s == "RIGHT":
        return HandSide.RIGHT
    if s == "LEFT":
        return HandSide.LEFT
    raise ValueError(f"hand_side must be RIGHT or LEFT, got: {s}")


def _extract_actuator_meta(env) -> dict:
    """Extract actuator names, joint names and ctrlrange from compiled mujoco model."""
    meta = {}
    phys = getattr(env, "physics", None)
    if phys is None:
        return meta

    m = phys.model
    nu = int(m.nu)
    meta["nu"] = nu
    meta["actuator_ctrlrange"] = np.asarray(m.actuator_ctrlrange).tolist()

    # actuator names
    act_names = []
    joint_names = []

    # actuator -> joint id via actuator_trnid[:,0]
    trnid = np.asarray(m.actuator_trnid)[:, 0]

    for i in range(nu):
        # actuator name
        try:
            act_names.append(m.actuator(i).name)
        except Exception:
            act_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

        # joint name
        jid = int(trnid[i])
        try:
            joint_names.append(m.joint(jid).name)
        except Exception:
            joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))

    meta["actuator_names"] = act_names
    meta["actuator_joint_names"] = joint_names
    return meta


def get_env(args: ExportArgs, record_dir: Optional[Path] = None):
    """
    Returns:
      env: wrapped env (canonical action space)
      tap: ActionTapWrapper sitting below CanonicalSpecWrapper, records physical actions
    """
    if args.midi_name not in library.MIDI_NAME_TO_CALLABLE:
        raise KeyError(
            f"Unknown midi_name={args.midi_name}. "
            f"Available: {list(library.MIDI_NAME_TO_CALLABLE.keys())}"
        )
    midi = library.MIDI_NAME_TO_CALLABLE[args.midi_name]()

    task = PianoWithOneShadowHand(
        midi=midi,
        hand_side=_parse_hand_side(args.hand_side),
        n_steps_lookahead=args.n_steps_lookahead,
        trim_silence=args.trim_silence,
        wrong_press_termination=args.wrong_press_termination,
        initial_buffer_time=args.initial_buffer_time,
        disable_fingering_reward=args.disable_fingering_reward,
        disable_colorization=args.disable_colorization,
        gravity_compensation=args.gravity_compensation,
        reduced_action_space=args.reduced_action_space,
        control_timestep=args.control_timestep,
        primitive_fingertip_collisions=args.primitive_fingertip_collisions,
        change_color_on_activation=True,
    )

    env = composer_utils.Environment(
        task=task,
        random_state=args.eval_seed,
        strip_singleton_obs_buffer_dim=True,
        legacy_step=True,
    )

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
    tap = ActionTapWrapper(env)  # tap sees *physical* action after de-normalization
    env = wrappers.CanonicalSpecWrapper(tap, clip=args.clip)

    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env, tap


def main(args: ExportArgs):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 用 record_dir 录制音视频 + 统计 + musical metrics
    record_dir = out_dir / "rollout"
    record_dir.mkdir(parents=True, exist_ok=True)

    env, tap = get_env(args, record_dir=record_dir)

    # 固定随机性：确保每次 reset 初始条件一致
    env.random_state.seed(args.eval_seed)

    # spec + agent init + restore
    spec = specs.EnvironmentSpec.make(env)
    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.eval_seed,
        discount=0.99,
    )
    agent = checkpoints.restore_checkpoint(
        ckpt_dir=args.ckpt_dir,
        target=agent,
    )

    # rollout（deterministic）
    ts = env.reset()

    actions_canonical = []
    actions_ctrl = []  # physical (de-normalized) action that actually goes to underlying env

    rewards = []
    discounts = []
    dones = []

    while True:
        if ts.last():
            break

        a = agent.eval_actions(ts.observation)  # deterministic, canonical (likely in [-1,1])
        actions_canonical.append(np.asarray(a))

        ts = env.step(a)

        # After env.step(a), CanonicalSpecWrapper will (likely) de-normalize a into physical action,
        # and ActionTapWrapper below will record it.
        if tap.last_action is None:
            raise RuntimeError(
                "ActionTapWrapper did not capture any action. "
                "Check CanonicalSpecWrapper behavior or wrapper order."
            )
        actions_ctrl.append(tap.last_action.copy())

        rewards.append(float(ts.reward))
        discounts.append(float(ts.discount))
        dones.append(bool(ts.last()))

    actions_canonical = np.asarray(actions_canonical)
    actions_ctrl = np.asarray(actions_ctrl)
    rewards = np.asarray(rewards, dtype=np.float32)
    discounts = np.asarray(discounts, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)

    # ---- Save files ----
    # 1) canonical actions (normalized)
    np.save(out_dir / "actions_canonical.npy", actions_canonical)
    # 2) physical ctrl actions (what underlying env actually executes)
    np.save(out_dir / "actions_ctrl.npy", actions_ctrl)
    # 3) For deployment convenience, make actions.npy = physical ctrl by default
    np.save(out_dir / "actions.npy", actions_ctrl)

    np.savez(
        out_dir / "traj.npz",
        actions_canonical=actions_canonical,
        actions_ctrl=actions_ctrl,
        rewards=rewards,
        discounts=discounts,
        dones=dones,
        eval_seed=np.int32(args.eval_seed),
    )

    # ---- Meta ----
    meta = {
        "ckpt_dir": str(args.ckpt_dir),
        "out_dir": str(out_dir),
        "eval_seed": int(args.eval_seed),
        "midi_name": args.midi_name,
        "hand_side": args.hand_side,
        "control_timestep": float(args.control_timestep),

        "action_dim_expected": int(spec.action_dim),

        "actions_canonical_shape": list(actions_canonical.shape),
        "actions_ctrl_shape": list(actions_ctrl.shape),

        "canonical_action_range_expected": [-1.0, 1.0],
        "actions_canonical_minmax": [float(actions_canonical.min()), float(actions_canonical.max())],
        "actions_ctrl_minmax": [float(actions_ctrl.min()), float(actions_ctrl.max())],

        # By default we now write actions.npy as physical ctrl:
        "actions_npy_is": "actions_ctrl",
    }

    # actuator mapping info (names/joints/ctrlrange)
    meta.update(_extract_actuator_meta(env))

    # wrappers stats / music metrics
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

    latest = getattr(env, "latest_filename", None)
    if latest is not None:
        meta["latest_video"] = str(latest)

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(tyro.cli(ExportArgs))
