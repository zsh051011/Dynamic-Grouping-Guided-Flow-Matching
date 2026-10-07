import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GroupAwareActorVectorField, GroupMixer, Value
from util import *
import distrax


class DyGFlowDiscreteAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()
    # Keep the fixed-shape assignments as a dynamic JAX leaf.  Their values
    # change during training and must not become part of the JIT cache key.
    group_ids: Any

    def _time_sin_embed(self, ts):
        """Sinusoidal time embedding shared across BC flow and integration."""
        kfreq = int(self.config.get('t_embed_frequencies', 8))
        freqs = jnp.asarray([2 ** i for i in range(kfreq)], dtype=ts.dtype) * jnp.pi
        ang = ts * freqs
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)

    # ---------------- Critic ---------------- #
    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        # Next actions from policy (indices) -> one-hot for Value network
        next_actions_idx = self.sample_actions_train(batch['observations'][1:], seed=sample_rng)
        next_actions = jax.nn.one_hot(next_actions_idx, self.config['action_dim'])

        # Target Q(s_{t+1}, a_{t+1}) across ensembles
        target_qs = self.network.select('target_q')(batch['observations'][1:], actions=next_actions)
        next_q = target_qs.min(axis=0) if self.config.get('q_agg', 'min') == 'min' else target_qs.mean(axis=0)

        # Current Q(s_t, a_t) using dataset actions (indices -> one-hot)
        cur_actions = jax.nn.one_hot(batch['actions'][:-1], self.config['action_dim'])
        qs_all = self.network.select('q')(batch['observations'][:-1], actions=cur_actions, params=grad_params)
        q_cur = qs_all.min(axis=0) if self.config.get('q_agg', 'min') == 'min' else qs_all.mean(axis=0)

        group_ids = jnp.asarray(self.group_ids, dtype=jnp.int32)
        mixed_next_q, _ = self.network.select('target_group_mixer')(
            next_q, batch['infos']['state'][1:], group_ids
        )
        mixed_q, importance = self.network.select('group_mixer')(
            q_cur, batch['infos']['state'][:-1], group_ids, params=grad_params
        )
        team_reward = jnp.sum(batch['rewards'][:-1], axis=-1)
        team_done = jnp.max(batch['terminals'][1:], axis=-1)
        mixed_target_q = team_reward + self.config['discount'] * (1.0 - team_done) * mixed_next_q
        critic_loss = 0.5 * jnp.mean(jnp.square(mixed_q - mixed_target_q))

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': mixed_q.mean(),
            'q_max': mixed_q.max(),
            'q_min': mixed_q.min(),
            'importance': importance.mean(axis=tuple(range(importance.ndim - 1))),
        }

    # ---------------- Actor (Flow Matching) ---------------- #
    @jax.jit
    def _build_actor_targets(self, batch, rng):
        action_dim = self.config['action_dim']
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        obs_slice = batch['observations'][:-1]
        act_slice = batch['actions'][:-1]
        x_1 = jax.nn.one_hot(act_slice, action_dim)
        x_0 = jax.random.normal(x_rng, (*x_1.shape[:-1], action_dim))
        t_scalar = jax.random.uniform(t_rng, (*x_1.shape[:-1], 1))
        x_t = (1 - t_scalar) * x_0 + t_scalar * x_1
        vel = x_1 - x_0
        t_embed = self._time_sin_embed(t_scalar)
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (*x_1.shape[:-1], action_dim))
        target_flow_actions = self.compute_flow_actions(
            obs_slice,
            noises=noises,
            is_encoded=self.config.get('use_lstm', False),
        )
        return {
            'x_t': x_t,
            'vel': vel,
            't_embed': t_embed,
            'noises': noises,
            'target_flow_actions': target_flow_actions,
        }

    def _actor_loss_from_targets(self, batch, grad_params, targets):
        action_dim = self.config['action_dim']
        obs_slice = batch['observations'][:-1]
        pred = self.network.select('actor_bc_flow')(
            obs_slice,
            targets['x_t'],
            jnp.asarray(self.group_ids, dtype=jnp.int32),
            targets['t_embed'],
            params=grad_params,
            is_encoded=self.config.get('use_lstm', False),
        )
        bc_flow_loss = jnp.mean((pred - targets['vel']) ** 2)

        actor_logits = self.network.select('actor_onestep_flow')(
            obs_slice,
            targets['noises'],
            jnp.asarray(self.group_ids, dtype=jnp.int32),
            params=grad_params,
            is_encoded=self.config.get('use_lstm', False),
        )
        distill_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                actor_logits, targets['target_flow_actions']
            )
        )

        # Q-guidance: prefer actions with higher mixed Q via Value(obs, onehot(action))
        actor_actions = jnp.argmax(actor_logits, axis=-1)
        actor_actions_oh = jax.nn.one_hot(actor_actions, action_dim)
        qs_all = self.network.select('q')(obs_slice, actions=actor_actions_oh)
        qs = qs_all.mean(axis=0)
        mixed_q_for_actor_actions, _ = self.network.select('group_mixer')(
            qs,
            batch['infos']['state'][:-1],
            jnp.asarray(self.group_ids, dtype=jnp.int32),
        )

        q_loss = -mixed_q_for_actor_actions.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(mixed_q_for_actor_actions).mean())
            q_loss = lam * q_loss

        # Apply the structural regularizer to policy logits so it trains only
        # the innovation module/policy path without changing the baseline LSTM
        # encoder's original gradient behaviour.
        group_loss = self._sd_loss(actor_logits)
        actor_loss = (
            bc_flow_loss
            + self.config['alpha'] * distill_loss
            + q_loss
            + self.config['sd_alpha'] * group_loss
        )

        # # Extra metric: MSE vs dataset actions under onestep policy
        # actions = self.sample_actions_train(batch['observations'][:-1], seed=rng)
        # mse = jnp.mean((actions - batch['actions'][:-1]) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': mixed_q_for_actor_actions.mean(),
            'group_loss': group_loss,
            # 'mse': mse,
        }

    def actor_loss(self, batch, grad_params, rng):
        targets = self._build_actor_targets(batch, rng)
        return self._actor_loss_from_targets(batch, grad_params, targets)

    def _prepare_training_batch(self, batch):
        observations = batch['observations']  # (B,T,N,O)
        actions = batch['actions']  # (B,T,N)
        env_states = batch['infos']['state']  # (B,T,S)
        rewards = batch['rewards']  # (B,T,N)
        terminals = jnp.logical_or(
            batch['terminals'], batch['truncations']
        ).astype(jnp.float32)

        observations = batch_concat_agent_id_to_obs(observations)

        obs_t = switch_two_leading_dims(observations)
        actions_t = switch_two_leading_dims(actions)
        rewards_t = switch_two_leading_dims(rewards)
        terminals_t = switch_two_leading_dims(terminals)

        if self.config.get('use_lstm', False):
            resets = jnp.zeros_like(terminals_t, dtype=jnp.bool_)
            resets = resets.at[0].set(True)
            resets = resets.at[1:].set(terminals_t[:-1] > 0.5)
            enc_obs_t = self.network.select('seq_encoder')(
                obs_t, resets
            )
            obs_in = enc_obs_t
        else:
            obs_in = obs_t

        return {
            'observations': obs_in,
            'actions': actions_t,
            'rewards': rewards_t,
            'terminals': terminals_t,
            'infos': {
                'state': switch_two_leading_dims(env_states)
            }
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        prepared_batch = self._prepare_training_batch(batch)

        critic_loss, critic_info = self.critic_loss(
            prepared_batch, grad_params, critic_rng
        )
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(
            prepared_batch, grad_params, actor_rng
        )
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def _sd_loss(self, features):
        """Encourage within-group similarity and between-group diversity."""
        x = features / (jnp.linalg.norm(features, axis=-1, keepdims=True) + 1e-6)
        similarity = jnp.einsum('...id,...jd->...ij', x, x)
        gids = jnp.asarray(self.group_ids, dtype=jnp.int32)
        same = gids[:, None] == gids[None, :]
        eye = jnp.eye(gids.shape[0], dtype=jnp.bool_)
        within = same & ~eye
        between = ~same
        within_loss = jnp.sum((1.0 - similarity) * within) / (
            jnp.sum(within) * jnp.prod(jnp.asarray(similarity.shape[:-2])) + 1e-6
        )
        between_loss = jnp.sum(jnp.square(jnp.maximum(similarity, 0.0)) * between) / (
            jnp.sum(between) * jnp.prod(jnp.asarray(similarity.shape[:-2])) + 1e-6
        )
        return within_loss + between_loss

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step):
        """Apply one full-batch update, matching the baseline execution path."""
        del step
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'q')
        self.target_update(new_network, 'group_mixer')
        return self.replace(network=new_network, rng=new_rng), info

    def update_group_assignments(self, importance):
        """Apply select-and-kick-out on the host at scheduled update steps."""
        values = jax.device_get(importance)
        groups = [
            int(group)
            for group in np.asarray(jax.device_get(self.group_ids)).tolist()
        ]
        next_group = max(groups) + 1
        spawned_groups = []
        for gid in sorted(set(groups)):
            members = [i for i, g in enumerate(groups) if g == gid]
            if len(members) <= 1:
                continue
            mean_value = float(sum(values[i] for i in members) / len(members))
            kicked = [
                i for i in members
                if float(values[i]) < mean_value * self.config['change_group_value']
            ]
            if len(kicked) == len(members):
                kicked.remove(max(members, key=lambda i: float(values[i])))
            if kicked:
                for i in kicked:
                    groups[i] = next_group
                spawned_groups.append((gid, next_group))
                next_group += 1
        remap = {old: new for new, old in enumerate(sorted(set(groups)))}
        compact = tuple(remap[g] for g in groups)
        remapped_spawns = [(remap[parent], remap[child]) for parent, child in spawned_groups]
        network = self._warm_start_group_slots(self.network, remapped_spawns)
        return self.replace(
            group_ids=jnp.asarray(compact, dtype=jnp.int32),
            network=network,
        )

    def _warm_start_group_slots(self, network, spawned_groups):
        """Copy parent group parameters into newly activated fixed slots.

        This runs only when host-side group assignments change.  Performing the
        tiny indexed updates with ``Array.at`` asks XLA to compile additional GPU
        kernels at that point in training.  Copy through NumPy instead so a
        group update cannot launch a late ptxas job; the copied parameter values
        are otherwise identical.
        """
        if not spawned_groups:
            return network

        def copy_group_slot(value, parent, child):
            device = value.device
            host_value = np.array(jax.device_get(value), copy=True)
            host_value[child] = host_value[parent]
            return jax.device_put(host_value, device)

        params = copy.deepcopy(network.params)
        for parent, child in spawned_groups:
            for module_name in ('modules_group_mixer', 'modules_target_group_mixer'):
                module_params = params[module_name]
                module_params['group_hyper'] = jax.tree_util.tree_map(
                    lambda value: copy_group_slot(value, parent, child),
                    module_params['group_hyper'],
                )
            for module_name in (
                'modules_actor_bc_flow',
                'modules_actor_onestep_flow',
            ):
                embeddings = params[module_name]['group_embeddings']
                params[module_name]['group_embeddings'] = copy_group_slot(
                    embeddings, parent, child
                )
        return network.replace(params=params)

    # ---------------- Acting ---------------- #
    def _masked_softmax(self, logits, mask=None):
        if mask is None:
            return jax.nn.softmax(logits, axis=-1)
        large_neg = -1e9
        masked_logits = jnp.where(mask, logits, large_neg)
        return jax.nn.softmax(masked_logits, axis=-1)

    @jax.jit
    def sample_actions_train(self, observations, seed=None, temperature=1.0, stochastic: bool = False):
        seed = self.rng if seed is None else seed
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(action_seed, (*observations.shape[:3], self.config['action_dim']))
        logits = self.network.select('actor_onestep_flow')(
            observations,
            noises,
            jnp.asarray(self.group_ids, dtype=jnp.int32),
            is_encoded=self.config.get('use_lstm', False),
        )
        logits = logits / jnp.maximum(temperature, 1e-6)
        probs = jax.nn.softmax(logits, axis=-1)
        if stochastic:
            flat = probs.reshape((-1, probs.shape[-1]))
            keys = jax.random.split(action_seed, flat.shape[0])
            samples = jax.vmap(lambda p, k: distrax.Categorical(probs=p).sample(seed=k))(flat, keys)
            return samples.reshape(probs.shape[:-1])
        return jnp.argmax(probs, axis=-1)

    def sample_actions_with_carry(self,
                                  observations: Dict[str, jnp.ndarray],
                                  carry,
                                  seed,
                                  legal_actions=None,
                                  reset_mask=None,
                                  temperature: float = 0.0,
                                  stochastic: bool = False):
        rng = seed if seed is not None else self.rng
        action_seed, _ = jax.random.split(rng)
        N = len(self.agent_names)
        obs_with_ids = [concat_agent_id_to_obs(observations[agent], i, N) for i, agent in enumerate(self.agent_names)]
        obs_tensor = jnp.stack(obs_with_ids, axis=0)

        if self.config.get('use_lstm', False):
            obs_seq = obs_tensor[None, None, ...]
            if reset_mask is None:
                resets = jnp.ones((1, 1, N), dtype=jnp.bool_) if carry is None else jnp.zeros((1, 1, N), dtype=jnp.bool_)
            else:
                resets = reset_mask[None, None, :].astype(jnp.bool_)
            enc, new_carry = self.network.select('seq_encoder')(
                obs_seq,
                resets,
                initial_carry=carry,
                return_carry=True,
            )
            obs_tensor = enc[0, 0]
        else:
            new_carry = None

        noises = jnp.zeros((N, self.config['action_dim']))
        logits = self.network.select('actor_onestep_flow')(
            obs_tensor[None, :],
            noises[None, :],
            jnp.asarray(self.group_ids, dtype=jnp.int32),
            is_encoded=self.config.get('use_lstm', False),
        )[0]
        logits = logits / jnp.maximum(temperature, 1e-6)

        if legal_actions is not None:
            masks = jnp.stack([legal_actions[agent].astype(bool) for agent in self.agent_names], axis=0)
            probs = self._masked_softmax(logits, masks)
        else:
            probs = jax.nn.softmax(logits, axis=-1)

        if stochastic:
            keys = jax.random.split(action_seed, N)
            actions = jax.vmap(lambda p, k: distrax.Categorical(probs=p).sample(seed=k))(probs, keys)
        else:
            actions = jnp.argmax(probs, axis=-1)
        return {agent: actions[i] for i, agent in enumerate(self.agent_names)}, new_carry

    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed, legal_actions=None, temperature=0.0, stochastic: bool = False):
        actions, _ = self.sample_actions_with_carry(
            observations,
            carry=None,
            seed=seed,
            legal_actions=legal_actions,
            reset_mask=None,
            temperature=temperature,
            stochastic=stochastic,
        )
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises, is_encoded=False):
        if (self.config['encoder'] is not None) and (not self.config.get('use_lstm', False)) and (not is_encoded):
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        steps = int(self.config['flow_steps'])

        def body(actions, i):
            t_scalar = jnp.full((*observations.shape[:-1], 1), i / steps)
            t_embed = self._time_sin_embed(t_scalar)
            vels = self.network.select('actor_bc_flow')(
                observations,
                actions,
                jnp.asarray(self.group_ids, dtype=jnp.int32),
                t_embed,
                is_encoded=True,
            )
            actions = actions + vels / steps
            return actions, None

        actions, _ = jax.lax.scan(body, noises, jnp.arange(steps))
        return jnp.argmax(actions, axis=-1)

    # ---------------- Factory ---------------- #
    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        agent_names,
        config,
    ):
        master_rng = jax.random.PRNGKey(seed)
        master_rng, big_init_rng = jax.random.split(master_rng, 2)

        T, B, N, O = ex_observations.shape
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        num_agents = len(agent_names)

        encoders = {}
        if (config['encoder'] is not None) and (not config.get('use_lstm', False)):
            encoder_module = encoder_modules[config['encoder']]
            encoders['q'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        q_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('q'),
        )
        actor_bc_flow_def = GroupAwareActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            num_agents=num_agents,
            attention_dim=config['group_attention_dim'],
            inter_group_top_k=config['inter_group_top_k'],
            decentralized_execution=config['decentralized_execution'],
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = GroupAwareActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            num_agents=num_agents,
            attention_dim=config['group_attention_dim'],
            inter_group_top_k=config['inter_group_top_k'],
            decentralized_execution=config['decentralized_execution'],
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )
        group_mixer_def = GroupMixer(
            state_dim=config['state_dim'],
            n_agents=num_agents,
            embed_dim=config['group_embed_dim'],
            hypernet_embed=config['group_hypernet_dim'],
        )

        ex_obs_with_id = batch_concat_agent_id_to_obs(
            ex_observations
        )
        ex_actions_init = ex_actions
        kfreq = int(config.get('t_embed_frequencies', 8))
        ex_times = jnp.zeros(
            (*ex_actions_init.shape[:-1], 2 * kfreq), dtype=jnp.float32
        )

        if config.get('use_lstm', False):
            enc_feat_dim = int(config.get('lstm_hidden_dim', 256))
            ex_obs_init = jnp.zeros(
                (*ex_obs_with_id.shape[:-1], enc_feat_dim), dtype=jnp.float32
            )
        else:
            ex_obs_init = ex_obs_with_id

        network_info = dict(
            q=(q_def, (ex_obs_init, ex_actions_init)),
            target_q=(copy.deepcopy(q_def), (ex_obs_init, ex_actions_init)),
            actor_bc_flow=(
                actor_bc_flow_def,
                (
                    ex_obs_init,
                    ex_actions_init,
                    jnp.zeros((num_agents,), dtype=jnp.int32),
                    ex_times,
                ),
            ),
            actor_onestep_flow=(
                actor_onestep_flow_def,
                (
                    ex_obs_init,
                    ex_actions_init,
                    jnp.zeros((num_agents,), dtype=jnp.int32),
                ),
            ),
            group_mixer=(
                group_mixer_def,
                (
                    jnp.zeros(ex_actions_init.shape[:-1], dtype=jnp.float32),
                    jnp.zeros(
                        (*ex_actions_init.shape[:2], config['state_dim']),
                        dtype=jnp.float32,
                    ),
                    jnp.zeros((num_agents,), dtype=jnp.int32),
                ),
            ),
            target_group_mixer=(
                copy.deepcopy(group_mixer_def),
                (
                    jnp.zeros(ex_actions_init.shape[:-1], dtype=jnp.float32),
                    jnp.zeros(
                        (*ex_actions_init.shape[:2], config['state_dim']),
                        dtype=jnp.float32,
                    ),
                    jnp.zeros((num_agents,), dtype=jnp.int32),
                ),
            ),
        )

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_obs_with_id,))

        if config.get('use_lstm', False):
            from utils.networks import SequenceLSTMEncoder  # type: ignore
            point_enc = None
            if config.get('encoder', None) is not None:
                point_enc = encoder_modules[config['encoder']]
            seq_enc_def = SequenceLSTMEncoder(
                hidden_dim=config.get('lstm_hidden_dim', 256),
                num_layers=config.get('lstm_layers', 1),
                pre_mlp_dims=tuple(config.get('lstm_pre_mlp_dims', ())),
                layer_norm=config.get('lstm_layer_norm', False),
                point_encoder=(point_enc() if point_enc is not None else None),
            )
            dummy_resets = jnp.zeros(
                ex_actions_init.shape[:-1], dtype=jnp.bool_
            )
            network_info['seq_encoder'] = (seq_enc_def, (ex_obs_with_id, dummy_resets))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(
            big_init_rng, **network_args
        )['params']
        network = TrainState.create(
            network_def, network_params, tx=network_tx
        )

        params = network.params
        params['modules_target_q'] = params['modules_q']
        params['modules_target_group_mixer'] = params['modules_group_mixer']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['num_agents'] = num_agents

        return cls(
            rng=master_rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
            group_ids=jnp.zeros((num_agents,), dtype=jnp.int32),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='dygflow',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            actor_hidden_dims=(256, 256, 256, 256),
            value_hidden_dims=(256, 256, 256, 256),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            alpha=3.,
            flow_steps=10,
            stop_grad_q_in_actor=True,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            discrete=True,
            q_agg='mean',
            t_embed_frequencies=8,
            # Sequence encoder (LSTM) options
            use_lstm=True,
            lstm_hidden_dim=64,
            lstm_layers=1,
            lstm_pre_mlp_dims=(128,),
            lstm_layer_norm=True,
            state_dim=ml_collections.config_dict.placeholder(int),
            group_embed_dim=32,
            group_hypernet_dim=64,
            sd_alpha=0.05,
            change_group_value=0.8,
            group_update_freq=10_000,
            group_attention_dim=64,
            inter_group_top_k=2,
            decentralized_execution=False,
        )
    )
