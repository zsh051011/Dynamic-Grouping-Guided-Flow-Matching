import json
import os
import random
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections import config_flags, config_dict

from envs.environments import get_environment
from utils.flax_utils import restore_agent
from utils.replay_buffers import FlashbaxReplayBuffer


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory containing params_<step>.pkl")
flags.DEFINE_integer("checkpoint_step", 2000000, "Checkpoint step to restore")
flags.DEFINE_integer("seed", 100, "Random seed")
flags.DEFINE_string("env", "smac_v1", "Environment name")
flags.DEFINE_string("source", "og_marl", "Dataset/environment source")
flags.DEFINE_string("scenario", "25m", "SMAC scenario")
flags.DEFINE_string("dataset", "Good", "Dataset tier")
flags.DEFINE_string("agent_name", "dygflow", "Agent name")
flags.DEFINE_string("data_dir", "./data", "Dataset directory")
flags.DEFINE_string("native_data_dir", "", "Local native SMAC npy dataset directory")
flags.DEFINE_integer("num_eval_episodes", 100, "Number of evaluation episodes")
flags.DEFINE_integer("sequence_length", 20, "Replay sequence length")
flags.DEFINE_integer("sample_period", 1, "Sample period")
flags.DEFINE_integer("buffer_size", 320000, "Replay buffer size")
flags.DEFINE_integer("batch_size", 8, "Batch size used for model shape initialization")
flags.DEFINE_string("output_json", "", "Output JSON path")

config_flags.DEFINE_config_file("agent", "agents/dygflow.py", lock_config=False)
flags.DEFINE_alias("agent_config", "agent")


def _evaluate(agent, env, n_eps=10, seed=0):
    episode_returns = []
    episode_wins = []
    episode_lengths = []
    episode_times = []
    invalid_action_count = 0
    action_count = 0
    noop_count = 0
    stop_count = 0
    move_count = 0
    attack_count = 0
    action_time_total = 0.0
    model_forward_calls = 0
    eval_start = time.time()

    for ep_idx in range(n_eps):
        observations, infos = env.reset()

        done = False
        episode_return = 0.0
        episode_steps = 0
        carry = None
        reset_mask = None
        episode_start = time.time()

        while not done:
            t_action = time.time()
            actions, carry = agent.sample_actions_with_carry(
                observations,
                carry,
                jax.random.PRNGKey(seed),
                infos.get("legals"),
                reset_mask=reset_mask,
            )
            action_time_total += time.time() - t_action
            model_forward_calls += 1

            actions = {
                agent_name: int(actions[agent_name].item())
                for agent_name in agent.agent_names
            }

            legal_actions = infos.get("legals")
            if legal_actions is not None:
                for agent_name, action in actions.items():
                    action_count += 1
                    legal = np.asarray(legal_actions[agent_name])
                    if action < 0 or action >= len(legal) or legal[action] < 0.5:
                        invalid_action_count += 1
                    if action == 0:
                        noop_count += 1
                    elif action == 1:
                        stop_count += 1
                    elif 2 <= action <= 5:
                        move_count += 1
                    else:
                        attack_count += 1

            observations, rewards, terminal, truncation, infos = env.step(actions)
            episode_return += np.mean(list(rewards.values()), dtype="float")
            episode_steps += 1
            reset_mask = jnp.asarray([terminal[ag] for ag in agent.agent_names], dtype=jnp.bool_)
            done = all(terminal.values()) or all(truncation.values())

        episode_returns.append(episode_return)
        episode_wins.append(float(infos.get("battle_won", False)))
        episode_lengths.append(episode_steps)
        episode_times.append(time.time() - episode_start)
        print(
            f"Episode ({ep_idx}): reward={episode_return:.6f} "
            f"battle_won={bool(infos.get('battle_won', False))} "
            f"length={episode_steps} "
            f"time_sec={episode_times[-1]:.2f}",
            flush=True,
        )

    eval_time = time.time() - eval_start
    total_env_steps = int(np.sum(episode_lengths))
    action_den = max(action_count, 1)
    return {
        "evaluation/mean_episode_return": float(np.mean(episode_returns)),
        "evaluation/max_episode_return": float(np.max(episode_returns)),
        "evaluation/min_episode_return": float(np.min(episode_returns)),
        "evaluation/std_episode_return": float(np.std(episode_returns)),
        "evaluation/win_rate": float(np.mean(episode_wins)),
        "evaluation/num_episodes": int(n_eps),
        "evaluation/total_env_steps": total_env_steps,
        "evaluation/avg_episode_length": float(np.mean(episode_lengths)),
        "evaluation/eval_time_sec": float(eval_time),
        "evaluation/avg_episode_time_sec": float(np.mean(episode_times)),
        "evaluation/env_steps_per_sec": float(total_env_steps / max(eval_time, 1e-8)),
        "evaluation/episodes_per_sec": float(n_eps / max(eval_time, 1e-8)),
        "evaluation/action_time_sec": float(action_time_total),
        "evaluation/avg_action_time_sec": float(action_time_total / max(model_forward_calls, 1)),
        "evaluation/model_forward_calls": int(model_forward_calls),
        "evaluation/invalid_action_count": int(invalid_action_count),
        "evaluation/invalid_action_rate": float(invalid_action_count / action_den),
        "evaluation/noop_action_rate": float(noop_count / action_den),
        "evaluation/stop_action_rate": float(stop_count / action_den),
        "evaluation/move_action_rate": float(move_count / action_den),
        "evaluation/attack_action_rate": float(attack_count / action_den),
        "evaluation/per_episode_return": episode_returns,
        "evaluation/per_episode_win": episode_wins,
        "evaluation/per_episode_length": episode_lengths,
        "evaluation/per_episode_time_sec": episode_times,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main(_):
    if FLAGS.checkpoint_dir is None:
        raise ValueError("--checkpoint_dir is required")

    agent_cfg = FLAGS.agent
    if hasattr(agent_cfg, "copy_and_resolve_references"):
        cfg = agent_cfg.copy_and_resolve_references()
    else:
        cfg = config_dict.ConfigDict(agent_cfg.to_dict())

    agent_name = FLAGS.agent_name
    cfg["agent_name"] = agent_name

    if agent_name == "dygflow":
        from agents.dygflow import DyGFlowDiscreteAgent

        agent_cls = DyGFlowDiscreteAgent
    elif agent_name == "macflow":
        from agents.discrete_macflow import MACFlowDiscreteAgent

        agent_cls = MACFlowDiscreteAgent
    else:
        raise ValueError(f"Unsupported agent_name for checkpoint eval: {FLAGS.agent_name}")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    env = get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, FLAGS.seed)
    agent_names = list(env.agents)

    buffer = FlashbaxReplayBuffer(
        sequence_length=FLAGS.sequence_length,
        batch_size=FLAGS.batch_size,
        sample_period=FLAGS.sample_period,
        seed=FLAGS.seed,
        max_size=FLAGS.buffer_size,
    )

    if FLAGS.native_data_dir:
        print(f"Loading native SMAC dataset from {FLAGS.native_data_dir}", flush=True)
        buffer.populate_from_native_npys(FLAGS.native_data_dir, agent_names)
    else:
        print(f"Loading local vault from {FLAGS.data_dir}", flush=True)
        buffer.populate_from_vault(
            FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset), rel_dir=FLAGS.data_dir
        )

    example = buffer.sample()
    ex_obs = jnp.asarray(example["observations"])
    ex_act = jnp.asarray(example["actions"])
    if len(ex_act.shape) == 3:
        ex_act = jnp.asarray(example["infos"]["legals"])
    cfg["state_dim"] = int(example["infos"]["state"].shape[-1])

    agent = agent_cls.create(
        seed=FLAGS.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        agent_names=agent_names,
        config=cfg,
    )
    agent = restore_agent(agent, FLAGS.checkpoint_dir, FLAGS.checkpoint_step)

    metrics = _evaluate(agent, env, n_eps=FLAGS.num_eval_episodes, seed=FLAGS.seed)
    metrics["evaluation/checkpoint_step"] = int(FLAGS.checkpoint_step)
    metrics["evaluation/dataset"] = str(FLAGS.dataset)
    metrics["evaluation/seed"] = int(FLAGS.seed)

    print(
        f"[eval] done checkpoint_step={FLAGS.checkpoint_step} "
        f"return={metrics['evaluation/mean_episode_return']:.6f} "
        f"win_rate={metrics['evaluation/win_rate']:.6f} "
        f"avg_len={metrics['evaluation/avg_episode_length']:.2f}",
        flush=True,
    )

    output_json = FLAGS.output_json
    if not output_json:
        output_json = os.path.join(
            FLAGS.checkpoint_dir, f"eval_final_{FLAGS.checkpoint_step}_{FLAGS.num_eval_episodes}.json"
        )
    with open(output_json, "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    print(f"[eval] wrote {output_json}", flush=True)


if __name__ == "__main__":
    app.run(main)
