from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution

class DiscreteActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2
    use_relaxed: bool = False
    encoder: Optional[nn.Module] = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logits_net = nn.Dense(self.action_dim,
                                   kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, temperature: float = 1.0):
        x = self.encoder(observations) if self.encoder else observations
        hidden = self.actor_net(x)
        logits = self.logits_net(hidden) / temperature

        if self.use_relaxed:
            distribution = distrax.RelaxedOneHotCategorical(
                temperature=temperature,
                logits=logits
            )
        else:
            distribution = distrax.Categorical(logits=logits)

        return distribution

class SoftmaxActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: Optional[nn.Module] = None

    def setup(self):
        mlp_class = MLP
        policy_net = mlp_class((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        self.policy_net = policy_net

    def __call__(self, observations):
        x = self.encoder(observations) if self.encoder else observations
        logits = self.policy_net(x)
        probs = nn.softmax(logits, axis=-1)
        return probs

class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v

class ActorVectorField(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class GroupAwareActorVectorField(nn.Module):
    """Flow vector field with hierarchical agent/group communication."""

    hidden_dims: Sequence[int]
    action_dim: int
    num_agents: int
    attention_dim: int = 64
    inter_group_top_k: int = 2
    decentralized_execution: bool = False
    layer_norm: bool = False
    encoder: nn.Module = None

    @nn.compact
    def __call__(
        self,
        observations,
        actions,
        group_ids,
        times=None,
        is_encoded=False,
    ):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        tokens = nn.Dense(self.attention_dim, name='token_projection')(observations)
        tokens = nn.LayerNorm(name='token_norm')(tokens)
        group_ids = jnp.asarray(group_ids, dtype=jnp.int32)
        membership = jax.nn.one_hot(
            group_ids, self.num_agents, dtype=tokens.dtype
        )
        group_embeddings = self.param(
            'group_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_agents, self.attention_dim),
        )
        tokens = tokens + jnp.einsum(
            'ag,gd->ad', membership, group_embeddings
        )

        # Dense communication between agents that currently share a group.
        q = nn.Dense(self.attention_dim, use_bias=False, name='within_q')(tokens)
        k = nn.Dense(self.attention_dim, use_bias=False, name='within_k')(tokens)
        v = nn.Dense(self.attention_dim, use_bias=False, name='within_v')(tokens)
        scores = jnp.einsum('...id,...jd->...ij', q, k)
        scores = scores / jnp.sqrt(jnp.asarray(self.attention_dim, tokens.dtype))
        if self.decentralized_execution:
            same_group = jnp.eye(self.num_agents, dtype=jnp.bool_)
        else:
            same_group = group_ids[:, None] == group_ids[None, :]
        scores = jnp.where(same_group, scores, -1e9)
        within_attention = jax.nn.softmax(scores, axis=-1)
        within_context = jnp.einsum('...ij,...jd->...id', within_attention, v)

        # Max-pool each group into a fixed n_agents-sized group token bank.
        member_mask = membership.astype(jnp.bool_)
        expanded_tokens = tokens[..., :, None, :]
        group_tokens = jnp.max(
            jnp.where(member_mask[..., None], expanded_tokens, -1e9),
            axis=-3,
        )
        active_groups = membership.sum(axis=0) > 0
        group_tokens = jnp.where(
            active_groups[..., None], group_tokens, jnp.zeros_like(group_tokens)
        )

        # Sparse global communication: each active group attends to top-k groups.
        group_q = nn.Dense(self.attention_dim, use_bias=False, name='group_q')(
            group_tokens
        )
        group_k = nn.Dense(self.attention_dim, use_bias=False, name='group_k')(
            group_tokens
        )
        group_v = nn.Dense(self.attention_dim, use_bias=False, name='group_v')(
            group_tokens
        )
        group_scores = jnp.einsum('...gd,...hd->...gh', group_q, group_k)
        group_scores = group_scores / jnp.sqrt(
            jnp.asarray(self.attention_dim, tokens.dtype)
        )
        group_eye = jnp.eye(self.num_agents, dtype=jnp.bool_)
        has_other_group = jnp.sum(active_groups) > 1
        valid_group_pairs = (
            active_groups[:, None]
            & active_groups[None, :]
            & ((~group_eye) | (~has_other_group))
        )
        group_scores = jnp.where(valid_group_pairs, group_scores, -1e9)
        top_k = max(1, min(int(self.inter_group_top_k), self.num_agents))
        _, top_indices = jax.lax.top_k(group_scores, top_k)
        sparse_mask = jnp.sum(
            jax.nn.one_hot(
                top_indices, self.num_agents, dtype=jnp.int32
            ),
            axis=-2,
        ).astype(jnp.bool_)
        sparse_mask = sparse_mask & valid_group_pairs
        group_scores = jnp.where(sparse_mask, group_scores, -1e9)
        group_attention = jax.nn.softmax(group_scores, axis=-1)
        group_attention = jnp.where(
            active_groups[..., None], group_attention, 0.0
        )
        group_context = jnp.einsum(
            '...gh,...hd->...gd', group_attention, group_v
        )
        agent_group_context = jnp.einsum(
            'ag,...gd->...ad', membership, group_context
        )
        if self.decentralized_execution:
            within_context = jnp.zeros_like(within_context)
            agent_group_context = jnp.zeros_like(agent_group_context)

        inputs = [
            observations,
            actions,
            within_context,
            agent_group_context,
        ]
        if times is not None:
            inputs.append(times)
        fused = jnp.concatenate(inputs, axis=-1)
        fused = nn.LayerNorm(name='fusion_norm')(fused)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
            name='policy_net',
        )(fused)

class DQN(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        return v

class QMixer(nn.Module):
    state_dim: int
    n_agents: int
    embed_dim: int
    hypernet_embed: int
    layer_norm: bool = True
    non_monotonic: bool = False

    def setup(self):
        mlp_class = MLP
        mixer_ = mlp_class((self.hypernet_embed,
                            self.embed_dim * self.n_agents),
                            activate_final=False, layer_norm=self.layer_norm)
        mixer_final = mlp_class((self.hypernet_embed, self.embed_dim), activate_final=False, layer_norm=self.layer_norm)
        b_ = nn.Dense(self.embed_dim, kernel_init=default_init())
        V_ = mlp_class((self.embed_dim, 1), activate_final=False, layer_norm=self.layer_norm)
        self.mixer_ = mixer_
        self.mixer_final = mixer_final
        self.b_ = b_
        self.V_ = V_

    def __call__(self, states, agent_qs=None):
        if agent_qs is not None:
            return self.b(agent_qs, states)
        else:
            return self.k(states)

    def b(self, agent_qs, states):
        B = agent_qs.shape[0]
        state_dim = states.shape[2:]
        agent_qs = jnp.reshape(agent_qs, (-1, 1, self.n_agents))
        states = jnp.reshape(states, (-1, *state_dim))

        w1 = self.mixer_(states)
        if not self.non_monotonic:
            w1 = jnp.abs(w1)
        b1 = self.b_(states)
        w1 = jnp.reshape(w1, (-1, self.n_agents, self.embed_dim))
        b1 = jnp.reshape(b1, (-1, 1, self.embed_dim))
        hidden = nn.elu(jnp.matmul(agent_qs, w1) + b1)

        w_final = self.mixer_final(states)
        if not self.non_monotonic:
            w_final = jnp.abs(w_final)
        w_final = jnp.reshape(w_final, (-1, self.embed_dim, 1))

        v = jnp.reshape(self.V_(states), (-1, 1, 1))
        y = jnp.matmul(hidden, w_final) + v
        q_tot = jnp.reshape(y, (B, -1, 1))
        return q_tot

    def k(self, states):
        B, T = states.shape[:2]

        w1 = jnp.abs(self.mixer_(states))
        w_final = jnp.abs(self.mixer_final(states))
        w1 = jnp.reshape(w1, (-1, self.n_agents, self.embed_dim))
        w_final = jnp.reshape(w_final, (-1, self.embed_dim, 1))
        k = jnp.matmul(w1, w_final)
        k = jnp.reshape(k, (B, -1, self.n_agents))
        k = k / (jnp.sum(k, axis=2, keepdims=True) + 1e-10)
        return k


class IndependentHyperNet(nn.Module):
    """Vectorised bank of independent two-layer hypernetworks."""

    num_heads: int
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        kernel1 = self.param(
            'kernel1',
            default_init(),
            (self.num_heads, input_dim, self.hidden_dim),
        )
        bias1 = self.param(
            'bias1',
            nn.initializers.zeros,
            (self.num_heads, self.hidden_dim),
        )
        hidden = jnp.einsum('bd,ndh->bnh', x, kernel1) + bias1
        hidden = nn.gelu(hidden)
        kernel2 = self.param(
            'kernel2',
            default_init(),
            (self.num_heads, self.hidden_dim, self.output_dim),
        )
        bias2 = self.param(
            'bias2',
            nn.initializers.zeros,
            (self.num_heads, self.output_dim),
        )
        return jnp.einsum('bnh,nho->bno', hidden, kernel2) + bias2


class GroupMixer(nn.Module):
    """State-conditioned monotonic mixer with a fixed-size dynamic group mask.

    Parameters are allocated for at most ``n_agents`` groups, while ``group_ids``
    decides which groups are active.  This keeps parameter shapes stable when a
    group is split during training.
    """

    state_dim: int
    n_agents: int
    embed_dim: int = 32
    hypernet_embed: int = 64

    @nn.compact
    def __call__(self, agent_qs, states, group_ids):
        leading = agent_qs.shape[:-1]
        flat_states = states.reshape((-1, states.shape[-1]))
        flat_states = nn.LayerNorm(name='state_norm')(flat_states)
        flat_qs = agent_qs.reshape((-1, self.n_agents))

        # Each head owns independent parameters; the vectorised implementation
        # avoids expanding 25 Python MLPs into a very large XLA graph.
        agent_hyper = IndependentHyperNet(
            num_heads=self.n_agents,
            hidden_dim=self.hypernet_embed,
            output_dim=2 * self.embed_dim,
            name='agent_hyper',
        )(flat_states)
        agent_w1_raw, agent_b1 = jnp.split(
            agent_hyper, 2, axis=-1
        )
        agent_w1 = nn.softplus(agent_w1_raw) + 1e-4
        agent_importance = jnp.mean(agent_w1, axis=-1)
        agent_importance = agent_importance / (
            jnp.mean(agent_importance, axis=-1, keepdims=True) + 1e-6
        )

        membership = jax.nn.one_hot(
            jnp.asarray(group_ids, dtype=jnp.int32),
            self.n_agents,
            dtype=flat_qs.dtype,
        )
        group_hidden = jnp.einsum(
            'bae,ag->bge',
            flat_qs[..., None] * agent_w1 + agent_b1,
            membership,
        )
        group_sizes = membership.sum(axis=0)
        group_hidden = group_hidden / jnp.maximum(
            group_sizes[None, :, None], 1.0
        )
        group_hidden = nn.elu(group_hidden)
        active = (group_sizes > 0).astype(flat_qs.dtype)

        group_hyper = IndependentHyperNet(
            num_heads=self.n_agents,
            hidden_dim=self.hypernet_embed,
            output_dim=self.embed_dim + 1,
            name='group_hyper',
        )(flat_states)
        group_weights = nn.softplus(group_hyper[..., :self.embed_dim]) + 1e-4
        group_bias = group_hyper[..., -1]
        group_values = (
            jnp.mean(group_hidden * group_weights, axis=-1)
            * group_sizes[None, :]
            + group_bias
        )
        mixed = jnp.sum(
            active[None, :] * group_values, axis=-1
        )

        return (
            mixed.reshape(leading),
            agent_importance.reshape((*leading, self.n_agents)),
        )

class MixNet(nn.Module):
    state_dim: int
    hidden_dims: int
    num_agents: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        self.f_v = mlp_class((self.state_dim, self.hidden_dims), activate_final=False,
                             layer_norm=self.layer_norm)
        self.w_v = mlp_class((self.hidden_dims, self.num_agents), activate_final=False,
                             layer_norm=self.layer_norm)
        self.b_v =mlp_class((self.hidden_dims, 1), activate_final=False,
                             layer_norm=self.layer_norm)

    def __call__(self, states):
        x = self.f_v(states)
        w = self.w_v(x)
        b = self.b_v(x)
        return jnp.abs(w), b



class SequenceLSTMEncoder(nn.Module):
    """Sequence encoder with optional per-step encoder and stacked LSTM.

    Inputs are expected to be time-major: (T, B, N, ...), but any leading batch
    shape is supported as long as time is the first dimension.

    - Optionally applies a `point_encoder` to each timestep independently
      (e.g., CNN/MLP) before the recurrent layers.
    - Stacked LSTM(s) then process features over time with reset masking.
    - Returns per-timestep hidden states of the top LSTM layer with the same
      leading shape as inputs, replacing the feature dimension.
    """

    hidden_dim: int
    num_layers: int = 1
    pre_mlp_dims: Sequence[int] = ()
    layer_norm: bool = False
    point_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, resets: Optional[jnp.ndarray] = None,
                 initial_carry: Optional[Sequence[Any]] = None,
                 return_carry: bool = False):
        # observations: (T, B, N, F...) -> flatten trailing feature dims if needed
        # Accept arbitrary leading dims after T; combine them to an effective batch M
        x = observations
        # Flatten any spatial dims before MLP/LSTM; apply point encoder first if given
        if self.point_encoder is not None:
            # point_encoder should broadcast over leading dims automatically
            x = self.point_encoder(x)
        # Ensure last dim is feature
        x = x.reshape(x.shape[: -1] + (-1,)) if x.ndim >= 2 else x

        T = x.shape[0]
        leading_shape = x.shape[1:-1]  # e.g., (B, N)
        # compute M using Python ints from shape, avoiding JAX tracers
        M = 1
        if len(leading_shape) > 0:
            for d in leading_shape:
                M *= d
        feat_dim = x.shape[-1]

        x_flat = x.reshape((T, M, feat_dim))

        if self.pre_mlp_dims:
            x_flat = MLP(self.pre_mlp_dims, activate_final=True, layer_norm=self.layer_norm)(x_flat)

        # Simple, version-agnostic zero carry with batch dim M
        def _zero_carry(batch_size: int, hidden: int, dtype):
            c = jnp.zeros((batch_size, hidden), dtype)
            h = jnp.zeros((batch_size, hidden), dtype)
            return (c, h)

        # Wrapper cell that applies reset mask before calling LSTMCell
        class _ResetLSTMCell(nn.Module):
            hidden_dim: int
            @nn.compact
            def __call__(self, carry, inputs):
                x_t, reset_t = inputs  # (M, D), (M,)
                # Avoid GPU QR in Flax's default orthogonal recurrent initializer,
                # which is unstable on this RTX 5080/JAX stack.
                lstm = nn.OptimizedLSTMCell(
                    self.hidden_dim,
                    recurrent_kernel_init=nn.initializers.lecun_normal(),
                )

                # Support both tuple and object state
                def _get_c_h(s):
                    if hasattr(s, 'c') and hasattr(s, 'h'):
                        return s.c, s.h
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        return s[0], s[1]
                    raise TypeError(f"Unrecognized LSTM state structure: {type(s)}")

                def _make_state_like(s, c, h):
                    if hasattr(s, 'c') and hasattr(s, 'h'):
                        cls = type(s)
                        try:
                            return cls(c=c, h=h)
                        except TypeError:
                            try:
                                return cls(c, h)
                            except TypeError:
                                return (c, h)
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        return (c, h)
                    return (c, h)

                c, h = _get_c_h(carry)
                c = jnp.where(reset_t[:, None], jnp.zeros_like(c), c)
                h = jnp.where(reset_t[:, None], jnp.zeros_like(h), h)
                carry = _make_state_like(carry, c, h)

                new_carry, y = lstm(carry, x_t)
                return new_carry, y

        if resets is None:
            resets_flat = jnp.zeros((T, M), dtype=bool)
            if initial_carry is None:
                resets_flat = resets_flat.at[0, :].set(True)
        else:
            assert resets.shape[: len(leading_shape) + 1] == (T, *leading_shape), (
                f"resets shape should be (T, *leading_shape), got {resets.shape}, expected {(T, *leading_shape)}"
            )
            resets_flat = resets.reshape((T, M)).astype(bool)
            if initial_carry is None:
                resets_flat = resets_flat.at[0, :].set(True)

        # Prepare initial carries
        if initial_carry is None:
            carries = tuple(_zero_carry(M, self.hidden_dim, x_flat.dtype) for _ in range(self.num_layers))
        else:
            carries = tuple(initial_carry)

        # Apply stacked LSTMs with nn.scan over time
        y = x_flat
        new_carries = []
        for i in range(self.num_layers):
            ScannedCell = nn.scan(
                _ResetLSTMCell,
                variable_broadcast=('params',),
                split_rngs={'params': False},
                in_axes=0,
                out_axes=0,
            )
            scan_cell = ScannedCell(self.hidden_dim, name=f'lstm_scan_{i}')
            carry_i, y = scan_cell(carries[i], (y, resets_flat))
            new_carries.append(carry_i)

        out = y.reshape((T, *leading_shape, self.hidden_dim))
        final_carry = tuple(new_carries)
        if return_carry:
            return out, final_carry
        return out
