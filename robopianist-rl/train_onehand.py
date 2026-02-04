"""
One-hand training script for RoboPianist-RL (SAC).
- Build env by directly instantiating PianoWithOneShadowHand (no suite.load).
- Save periodic checkpoints and best checkpoint/video by a chosen metric.
"""

from pathlib import Path
from typing import Optional, Tuple

import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import json
import shutil
from flax.training import checkpoints

import sac
import specs
import replay

import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

from mujoco_utils import composer_utils
from robopianist.music import library
from robopianist.models.hands import HandSide
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand



@dataclass(frozen=True)
class Args:
    # ===== basic =====
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    batch_size: int = 256
    discount: float = 0.99
    replay_capacity: int = 1_000_000

    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    tqdm_bar: bool = False

    # ===== wandb =====
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "offline"  # "online"/"offline"/"disabled"

    # ===== one-hand env =====
    midi_name: str = "TwinkleTwinkleLittleStar"  # key in library.MIDI_NAME_TO_CALLABLE
    hand_side: str = "RIGHT"  # "RIGHT" or "LEFT"

    n_steps_lookahead: int = 10
    trim_silence: bool = False
    initial_buffer_time: float = 0.5  # 给手一点时间移动到第一个音符附近

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
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"

    # agent
    agent_config: sac.SACConfig = sac.SACConfig()

    # ===== checkpoint / best =====
    checkpoint_interval: int = 200_000   # 每隔多少步保存一次训练断点
    checkpoint_keep: int = 3             # 最多保留多少个断点

    best_metric: str = "f1"              # 用哪个指标判定更好
    best_min_delta: float = 0.01         # 至少提升多少才算“更好”
    save_best_video: bool = True
    save_best_ckpt: bool = True


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def _parse_hand_side(s: str) -> HandSide:
    s = s.upper()
    if s == "RIGHT":
        return HandSide.RIGHT
    if s == "LEFT":
        return HandSide.LEFT
    raise ValueError(f"hand_side must be RIGHT or LEFT, got: {s}")


def get_env(args: Args, record_dir: Optional[Path] = None):
    # 1) build midi
    if args.midi_name not in library.MIDI_NAME_TO_CALLABLE:
        raise KeyError(
            f"Unknown midi_name={args.midi_name}. "
            f"Available: {list(library.MIDI_NAME_TO_CALLABLE.keys())}"
        )
    midi = library.MIDI_NAME_TO_CALLABLE[args.midi_name]()

    # 2) build one-hand task
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

    # 3) env
    env = composer_utils.Environment(
        task=task,
        random_state=args.seed,
        strip_singleton_obs_buffer_dim=True,
        legacy_step=True,
    )

    # 4) wrappers (保持你原来的逻辑)
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

    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


def main(args: Args) -> None:
    # run name
    run_name = args.name if args.name else f"SAC-onehand-{args.midi_name}-{args.hand_side}-{args.seed}-{time.time():.0f}"

    # dirs
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = experiment_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dir = experiment_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_state_path = best_dir / "best.json"
    best_score = -float("inf")
    if best_state_path.exists():
        try:
            best_score = float(json.loads(best_state_path.read_text()).get("score", best_score))
        except Exception:
            pass

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # wandb
    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )

    # envs
    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval")

    spec = specs.EnvironmentSpec.make(env)

    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )

    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()

    for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        # act
        if i < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        # step
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        # episode reset
        if timestep.last():
            wandb.log(prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # train
        if i >= args.warmstart_steps and replay_buffer.is_ready():
            transitions = replay_buffer.sample()
            agent, metrics = agent.update(transitions)
            if i % args.log_interval == 0:
                wandb.log(prefix_dict("train", metrics), step=i)

        # fps
        if i % args.log_interval == 0:
            wandb.log({"train/fps": int(i / (time.time() - start_time + 1e-6))}, step=i)
        
        # periodic checkpoint (independent of eval)
        if args.checkpoint_interval > 0 and (i % args.checkpoint_interval == 0):
            checkpoints.save_checkpoint(
                ckpt_dir=str(ckpt_dir),
                target=agent,
                step=i,
                keep=args.checkpoint_keep,
                overwrite=True,
            )


        # eval
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                ts = eval_env.reset()
                while not ts.last():
                    ts = eval_env.step(agent.eval_actions(ts.observation))

            stats = eval_env.get_statistics()
            music = eval_env.get_musical_metrics()

            log_dict = prefix_dict("eval", stats)
            music_dict = prefix_dict("eval", music)

            # pick score
            metric_candidates = [
                args.best_metric,
                "f1", "note_f1", "frame_f1", "precision_recall_f1", "key_f1",
            ]
            score = None
            for k in metric_candidates:
                if k in music:
                    score = float(music[k])
                    break

            wandb.log(log_dict | music_dict, step=i)

            # save video to wandb if exists
            latest = getattr(eval_env, "latest_filename", None)
            if latest is not None and Path(latest).exists():
                try:
                    video = wandb.Video(str(latest), fps=4, format="mp4")
                    wandb.log({"video": video, "global_step": i}, step=i)
                except Exception:
                    pass


            # best selection
            improved = (score is not None) and (score > best_score + args.best_min_delta)
            if improved:
                best_score = score
                meta = {"step": int(i), "score": float(best_score), "metric": args.best_metric}

                # best ckpt
                if args.save_best_ckpt:
                    best_ckpt_dir = best_dir / "checkpoint_best"
                    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    checkpoints.save_checkpoint(
                        ckpt_dir=str(best_ckpt_dir),
                        target=agent,
                        step=i,
                        keep=1,
                        overwrite=True,
                    )
                    meta["best_ckpt_dir"] = str(best_ckpt_dir)

                # best video
                if args.save_best_video and latest is not None and Path(latest).exists():
                    dst = best_dir / f"best_step_{i}_score_{best_score:.6f}.mp4"
                    shutil.copy2(latest, dst)
                    meta["best_video"] = str(dst)

                best_state_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

            # cleanup eval video (keep only best copy)
            if latest is not None:
                try:
                    p = Path(latest)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    main(tyro.cli(Args))
