# multiagent_main.py
import importlib.util
import os, json, time, random
import concurrent.futures

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp, numpy as np, tqdm, wandb
from absl import app, flags
from ml_collections import config_flags, config_dict

from envs.environments import get_environment
from utils.replay_buffers import FlashbaxReplayBuffer
from vault_utils.download_vault import download_and_unzip_vault
from utils.loggers import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    setup_wandb,
    get_wandb_video,
)
from utils.flax_utils import save_agent

FLAGS = flags.FLAGS
# ------------------------------------- 기본 플래그 -------------------------------------------------
flags.DEFINE_string ('run_group',       'Debug', 'Run group')
flags.DEFINE_integer('seed',            0,       'Random seed')
flags.DEFINE_string ('env',             'smac_v1', 'env')
flags.DEFINE_string ('source',          'og_marl', 'builder')
flags.DEFINE_string ('scenario',        '3m', 'scenario')
flags.DEFINE_string ('dataset',         'Good',  'quality')
flags.DEFINE_string ('save_dir',        'exp/',    'Save directory')
flags.DEFINE_string('agent_name',       'dygflow',      'Agent name')
flags.DEFINE_string('project_name',       'test',   'WandB project name')
flags.DEFINE_string('data_dir',       './data/',    'Directory to store datasets')
flags.DEFINE_string('native_data_dir', '', 'Local native SMAC npy dataset directory')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of SMAC evaluation episodes')

flags.DEFINE_integer('offline_steps',    1_000_000, 'Offline gradient steps')
flags.DEFINE_integer('sequence_length',  20,        'Replay sequence length')
flags.DEFINE_integer('sample_period',  1,        'Sample period')
flags.DEFINE_integer('buffer_size',      2_000_000, 'Max transitions in buffer')
flags.DEFINE_integer('batch_size',      32, 'Batch size')

flags.DEFINE_integer('log_interval',  50_000,   '')
flags.DEFINE_integer('eval_interval', 100_000,  '')
flags.DEFINE_integer('save_interval', 1_000_001, '')
flags.DEFINE_integer('eval_video_episodes', 1, 'Number of evaluation episodes to record as video.')
flags.DEFINE_integer('eval_video_frame_skip', 1, 'Render every k-th frame when recording evaluation videos.')
flags.DEFINE_integer('num_eval_workers', 10, 'Number of parallel workers for evaluation (threads).')
flags.DEFINE_bool('eval_return_only', False, 'Only save return-related evaluation metrics.')
# -----------------------------------------------------------------------------------------------

config_flags.DEFINE_config_file('agent', 'agents/dygflow.py', lock_config=False)
flags.DEFINE_alias('agent_config', 'agent')

def main(_):
    # Initialise and report the default backend before SMAC/WandB can import
    # libraries that also interact with CUDA.
    print(
        f"[jax] backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )
    agent_cfg = FLAGS.agent
    if hasattr(agent_cfg, 'copy_and_resolve_references'):
        cfg = agent_cfg.copy_and_resolve_references()
    else:
        cfg = config_dict.ConfigDict(agent_cfg.to_dict())
    requested_agent_name = FLAGS.agent_name
    agent_name = requested_agent_name
    cfg['agent_name'] = agent_name

    if agent_name == 'dygflow':
        from agents.dygflow import DyGFlowDiscreteAgent
        agent_cls = DyGFlowDiscreteAgent
    elif agent_name == 'macflow':
        from agents.discrete_macflow import MACFlowDiscreteAgent
        agent_cls = MACFlowDiscreteAgent
    else:
        raise ValueError(f"Unknown agent_name '{requested_agent_name}'. Please add it to AGENT_CONFIG_MAP and import here.")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=exp_name)

    save_root = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(save_root, exist_ok=True)
    json.dump(get_flag_dict(), open(os.path.join(save_root, 'flags.json'), 'w'))

    # -------- ENV & Buffer ----------------------------------- --------------------------
    env_factory = lambda s: get_environment(FLAGS.source, FLAGS.env, FLAGS.scenario, s)
    env = env_factory(FLAGS.seed)
    agent_names = list(env.agents)

    buffer = FlashbaxReplayBuffer(
        sequence_length=FLAGS.sequence_length,
        batch_size=FLAGS.batch_size,
        sample_period=FLAGS.sample_period,
        seed=FLAGS.seed,
        max_size=FLAGS.buffer_size)

    if FLAGS.native_data_dir:
        print(f"Loading native SMAC dataset from {FLAGS.native_data_dir}")
        buffer.populate_from_native_npys(FLAGS.native_data_dir, agent_names)
    elif FLAGS.data_dir == '/datastor1/dongsu':
        download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario,
                                 dataset_base_dir=FLAGS.data_dir)
        buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset),
                                   rel_dir=FLAGS.data_dir)
    else:
        vault_path = os.path.join(FLAGS.data_dir, FLAGS.source, FLAGS.env, f"{FLAGS.scenario}.vlt")
        if os.path.isdir(vault_path):
            print(f"Loading local vault from {vault_path}")
            buffer.populate_from_vault(
                FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset), rel_dir=FLAGS.data_dir)
        else:
            download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario)
            buffer.populate_from_vault(FLAGS.source, FLAGS.env, FLAGS.scenario, str(FLAGS.dataset))

    example = buffer.sample()
    ex_obs  = jnp.asarray(example['observations'])
    ex_act = jnp.asarray(example['actions'])
    if len(ex_act.shape) == 3:
        ex_act = jnp.asarray(example["infos"]["legals"])
    ex_state = jnp.asarray(example['infos']['state'])
    cfg['state_dim'] = int(ex_state.shape[-1])

    agent = agent_cls.create(
        seed=FLAGS.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        agent_names=agent_names,
        config=cfg,
    )

    # -------- Logger -----------------------------------------------------------------------
    train_csv = CsvLogger(os.path.join(save_root, 'train.csv'))
    eval_csv  = CsvLogger(os.path.join(save_root, 'eval.csv'))
    t0 = time.time(); last = t0

    importance_sum = np.zeros(len(agent_names), dtype=np.float64)
    importance_count = 0
    group_history_path = os.path.join(save_root, 'group_history.jsonl')
    if agent_name == 'dygflow':
        initial_groups = np.asarray(jax.device_get(agent.group_ids)).tolist()
        with open(group_history_path, 'w') as f:
            f.write(json.dumps({'step': 0, 'group_ids': initial_groups}) + '\n')
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), dynamic_ncols=True):
        if step <= FLAGS.offline_steps:
            batch = buffer.sample()

        agent, info = agent.update(batch, step)
        importance = info.pop('critic/importance', None)
        if importance is not None:
            importance_sum += np.asarray(jax.device_get(importance), dtype=np.float64)
            importance_count += 1
        update_freq = int(cfg.get('group_update_freq', 0))
        if (
            agent_name == 'dygflow'
            and update_freq > 0
            and step % update_freq == 0
            and importance_count > 0
        ):
            old_groups = tuple(
                np.asarray(jax.device_get(agent.group_ids)).tolist()
            )
            agent = agent.update_group_assignments(importance_sum / importance_count)
            new_groups = tuple(
                np.asarray(jax.device_get(agent.group_ids)).tolist()
            )
            print(
                f"[groups] step={step} old={old_groups} new={new_groups}",
                flush=True,
            )
            with open(group_history_path, 'a') as f:
                f.write(json.dumps({
                    'step': step,
                    'group_ids': list(new_groups),
                    'importance': (importance_sum / importance_count).tolist(),
                }) + '\n')
            importance_sum.fill(0.0)
            importance_count = 0

        if step % FLAGS.log_interval == 0:
            metrics = {f'train/{k}': float(v) for k, v in info.items()}
            if agent_name == 'dygflow':
                _, group_sizes = np.unique(agent.group_ids, return_counts=True)
                metrics['train/groups/count'] = float(len(group_sizes))
                metrics['train/groups/min_size'] = float(group_sizes.min())
                metrics['train/groups/max_size'] = float(group_sizes.max())
            metrics['time/iter_s'] = (time.time() - last) / FLAGS.log_interval
            print(
                f"[train] step={step}/{FLAGS.offline_steps} "
                f"iter_s={metrics['time/iter_s']:.6f} "
                + " ".join(f"{k}={v:.6f}" for k, v in metrics.items() if k != 'time/iter_s'),
                flush=True,
            )
            wandb.log(metrics, step=step)
            train_csv.log(metrics, step=step)
            last = time.time()

        if step % FLAGS.save_interval == 0:
            ckpt_path = os.path.join(save_root, f'params_{step}.pkl')
            save_agent(agent, save_root, step)
            print(f"[save] step={step} path={ckpt_path}", flush=True)

        if FLAGS.eval_interval > 0 and step % FLAGS.eval_interval == 0:
            print(f"[eval] start step={step} episodes={FLAGS.num_eval_episodes}", flush=True)
            try:
                eval_ret = _evaluate(agent, env, n_eps=FLAGS.num_eval_episodes, seed=FLAGS.seed)
            except Exception as exc:
                error_path = os.path.join(save_root, f'eval_step_{step}_error.txt')
                with open(error_path, 'w') as f:
                    f.write(repr(exc) + '\n')
                print(f"[eval] failed step={step} error={repr(exc)} error_path={error_path}", flush=True)
                raise
            print(
                f"[eval] done step={step} "
                f"return={eval_ret['evaluation/mean_episode_return']:.6f} "
                + (
                    ""
                    if FLAGS.eval_return_only
                    else (
                        f"win_rate={eval_ret['evaluation/win_rate']:.6f} "
                        f"avg_len={eval_ret['evaluation/avg_episode_length']:.2f} "
                        f"env_steps_per_sec={eval_ret['evaluation/env_steps_per_sec']:.2f}"
                    )
                ),
                flush=True,
            )
            if FLAGS.eval_return_only:
                eval_ret = _return_only_metrics(eval_ret)
            scalar_eval_ret = _scalar_metrics(eval_ret)
            wandb.log(scalar_eval_ret, step=step)
            eval_csv.log(scalar_eval_ret, step=step)
            with open(os.path.join(save_root, f'eval_step_{step}.json'), 'w') as f:
                json.dump(_json_safe(eval_ret), f, indent=2)

    train_csv.close(); eval_csv.close()

# === Helper ======================================================================================
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
            # Use stateful LSTM carry when enabled; reset at episode start or per-agent terminal
            actions, carry = agent.sample_actions_with_carry(
                observations,
                carry,
                jax.random.PRNGKey(seed),
                infos.get('legals'),
                reset_mask=reset_mask,
            )
            action_time_total += time.time() - t_action
            model_forward_calls += 1

            actions = {
                agent_name: int(actions[agent_name].item())
                for agent_name in agent.agent_names
            }

            legal_actions = infos.get('legals')
            if legal_actions is not None:
                for agent_name, action in actions.items():
                    action_count += 1
                    legal = np.asarray(legal_actions[agent_name])
                    if action < 0 or action >= len(legal) or legal[action] < 0.5:
                        invalid_action_count += 1
                        valid = np.flatnonzero(legal > 0.5)
                        actions[agent_name] = int(valid[0]) if len(valid) else 0
                        action = actions[agent_name]
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

            # Build per-agent reset mask for next step's RNN carry, if any agent terminated
            reset_mask = jnp.asarray([terminal[ag] for ag in agent.agent_names], dtype=jnp.bool_)
            done = all(terminal.values()) or all(truncation.values())

        episode_returns.append(episode_return)
        episode_wins.append(float(infos.get('battle_won', False)))
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
    logs = {
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
    return logs


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


def _return_only_metrics(metrics):
    keys = (
        "evaluation/mean_episode_return",
        "evaluation/max_episode_return",
        "evaluation/min_episode_return",
        "evaluation/std_episode_return",
        "evaluation/num_episodes",
        "evaluation/per_episode_return",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _scalar_metrics(metrics):
    scalar_types = (int, float, np.integer, np.floating)
    return {k: float(v) if isinstance(v, scalar_types) else v for k, v in metrics.items()
            if isinstance(v, scalar_types)}

# ===============================================================================================

if __name__ == '__main__':
    app.run(main)
