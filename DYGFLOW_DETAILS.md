# Dynamic Grouping-Guided Flow Matching (DyGFlow)

DyGFlow is an extension of MAC-Flow for offline multi-agent reinforcement
learning. The original MAC-Flow implementation is kept as the baseline, while
the new method is implemented in `agents/dygflow.py`.

## Method overview

DyGFlow adds the following components to MAC-Flow:

- `GroupMixer`, which uses the global state to generate per-agent importance
  scores and group-level mixing weights while preserving monotonicity.
- Dynamic grouping, initialized with one group and updated every
  `group_update_freq` steps using the mean importance over the update interval.
- A hierarchical flow actor with dense within-group attention, max-pooled group
  tokens, and top-k inter-group attention.
- A structural differentiation loss that aligns policy logits within a group
  and suppresses positive cosine similarity between different groups.
- Group-aware training and inference. Flow ODE integration, one-step policy
  inference, and checkpoint evaluation all use the same group assignments.
- Centralized and decentralized execution modes.
- A memory-mapped native SMAC NPY reader with episode-safe sequence sampling,
  shape validation, `filled` validation, and terminal/truncation merging.

Descriptions of MADIFF U-Net architectures and diffusion sampling do not map
directly onto MAC-Flow. DyGFlow therefore applies hierarchical agent/group
attention before the flow vector field and uses consistent group conditioning
for multi-step flow ODE integration and the distilled one-step policy.

## Environment used for development

The implementation was tested on an RTX 5080 with CUDA 12.8, cuDNN 9.10, and a
570-series NVIDIA driver. The development Conda environment used JAX 0.6.2.

The CUDA runtime components must come from one compatible installation. The
following settings were used during development:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu
export PATH=/usr/local/cuda-12.8/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

These paths are machine-specific and should be adjusted for the target system.

## Example: SMAC 25m

Run the command from the repository root and replace the dataset path with the
location of the native SMAC arrays:

```bash
python smac_main.py \
  --env=smac_v1 \
  --source=og_marl \
  --scenario=25m \
  --dataset=Good \
  --agent_name=dygflow \
  --native_data_dir=/path/to/native_smac/25m/Good \
  --offline_steps=400000 \
  --batch_size=16 \
  --sequence_length=20 \
  --agent.sd_alpha=0.05 \
  --agent.change_group_value=0.8 \
  --agent.group_update_freq=10000 \
  --agent.group_attention_dim=64 \
  --agent.inter_group_top_k=2 \
  --agent.decentralized_execution=false
```

Use `--offline_steps=2 --eval_interval=0 --save_interval=2` for an initial smoke
test. For a controlled comparison, keep the dataset, seed, training steps,
batch size, and number of evaluation episodes fixed, and run the original
MAC-Flow baseline under the same conditions.

The training directory stores grouping changes in `group_history.jsonl`.

## Scope of the comparison

DyGFlow preserves the original MAC-Flow Q-guidance `argmax` operation and LSTM
encoder call pattern. The architectural changes are limited to `GroupMixer`,
hierarchical attention, structural differentiation loss, dynamic grouping, and
group-aware inference.

## Implementation details

### 1. Training entry point

`smac_main.py` uses `DyGFlowDiscreteAgent` from `agents/dygflow.py` by default.
After sampling an example batch, it derives `state_dim` from
`example["infos"]["state"].shape[-1]`.

The training loop additionally maintains:

```text
importance_sum
importance_count
group_history.jsonl
```

Each batch produces per-agent importance values. At each grouping update, the
agent uses the mean importance accumulated over the complete interval rather
than the importance from a single batch. The logger also records:

```text
train/groups/count
train/groups/min_size
train/groups/max_size
```

Per-agent and per-group hypernetwork parameters remain independent. They are
stored as tensors with a head dimension and evaluated with `einsum`, which is
mathematically equivalent to expanding separate Python MLPs while producing a
smaller JAX/XLA compilation graph.

The complete `batch_size=16` loss is computed in one JIT-compiled forward and
backward pass followed by one Adam update. Sequence LSTMs, the ten-step teacher
flow, and all training modules run on the GPU. The teacher flow uses
`lax.scan`, and sparse inter-group attention uses `jax.lax.top_k`.

### 2. Two-stage GroupMixer

Implementation: `utils/networks.py::GroupMixer`.

The original discrete MAC-Flow critic uses a VDN-style sum:

```text
Q_tot = sum_i Q_i
```

DyGFlow first constructs an independent hypernetwork for each agent:

```text
w1_i = softplus(hyper_w1_i(state))
b1_i = hyper_b1_i(state)
```

Agent contributions are aggregated into group hidden states according to the
current `group_ids`:

```text
h_g = ELU(mean_{i in g}(Q_i * w1_i + b1_i))
```

Each potential group slot has its own second-stage hypernetwork:

```text
w2_g = softplus(hyper_w2_g(state))
b2_g = hyper_b2_g(state)
Q_g = |g| * mean(h_g * w2_g) + b2_g
Q_tot = sum_{active g} Q_g
```

`softplus` keeps the mixing weights positive and preserves monotonicity from
individual Q-values to the joint Q-value. The group hidden state is normalized
by group size and rescaled at the output to prevent hypernetwork biases from
growing excessively with the number of agents. The global state is normalized
with LayerNorm before entering the hypernetworks.

The online and target GroupMixers use the same `group_ids`. Target parameters
are softly updated using the original MAC-Flow `tau` setting.

### 3. Hierarchical group-aware flow actor

Implementation: `utils/networks.py::GroupAwareActorVectorField`.

The original `ActorVectorField` applies an independent MLP to each agent.
DyGFlow projects observations or LSTM features into agent tokens and applies
three stages of processing.

First, dense within-group attention is computed:

```text
mask[i,j] = group_ids[i] == group_ids[j]
A_intra = softmax(QK^T / sqrt(d), mask)
H_intra = A_intra V
```

Second, max pooling produces a fixed-capacity group token:

```text
H_group[g] = max_{i in g} H_agent[i]
```

Third, each active group attends only to its `inter_group_top_k` most similar
active groups:

```text
A_inter = top-k-softmax(Q_group K_group^T)
H_inter = A_inter V_group
```

The policy MLP receives local observations, flow action or noise, within-group
context, inter-group context, and the flow-time embedding on the behavior
cloning flow path. The same module is used for flow-matching velocity
prediction, multi-step ODE integration, policy distillation, and online action
generation.

### 4. Centralized and decentralized execution

The behavior is controlled by:

```text
--agent.decentralized_execution=false
```

Centralized mode enables within-group and inter-group context. In decentralized
mode, the within-group mask becomes the identity matrix, cross-agent context is
zeroed, and each agent retains only a learnable embedding selected by its group
identifier. Therefore, decentralized actions do not read other agents' current
observations while different groups can still learn differentiated policies.

### 5. Structural differentiation loss

Implementation: `agents/dygflow.py::_sd_loss`.

The loss operates on the one-step flow policy logits. After L2 normalization,
it computes pairwise cosine similarities between agents.

```text
L_within = mean_{same group, i != j}(1 - cosine(z_i, z_j))
L_between = mean_{different groups}(max(cosine(z_i, z_j), 0)^2)
```

The complete actor loss is:

```text
L_actor =
    L_BC_flow
    + alpha * L_distill
    + L_Q
    + sd_alpha * (L_within + L_between)
```

Applying the loss to logits preserves the baseline LSTM gradient path while
allowing the regularizer to optimize the policy directly.

### 6. Periodic select-and-kick-out grouping

Implementation: `agents/dygflow.py::update_group_assignments`.

All agents initially belong to one group:

```text
[0, 0, ..., 0]
```

The mean of the first-stage GroupMixer weights is normalized to obtain each
agent's importance. For group `g`, the threshold is:

```text
threshold_g = mean_importance_g * change_group_value
```

Members below the threshold move into a new group. The implementation keeps at
least the highest-importance member in every source group and compacts group
identifiers after each update so that no empty group remains.

### 7. Warm-starting new groups

Implementation: `agents/dygflow.py::_warm_start_group_slots`.

JAX and Flax require stable parameter-tree shapes, so the model allocates up to
`num_agents` group slots during initialization. Before an inactive slot is
activated, it receives copies of the parent group's parameters:

```text
group_hyper_w2
group_hyper_b2
behavior-cloning flow actor group embedding
one-step flow actor group embedding
corresponding target GroupMixer parameters
```

The new group therefore starts from its parent's behavior and differentiates
through subsequent optimization instead of activating random parameters.

### 8. Checkpoint format

Implementation: `utils/flax_utils.py`.

Because `group_ids` is not part of the standard parameter state, checkpoints
store it explicitly:

```python
save_dict["group_ids"] = tuple(agent.group_ids)
```

Checkpoint restoration reloads the network parameters and group assignments.
`eval_smac_checkpoint.py` constructs `DyGFlowDiscreteAgent`, ensuring that
evaluation uses the group structure learned at the end of training.

### 9. Native NPY replay reader

Implementation: `utils/replay_buffers.py::populate_from_native_npys`.

The reader memory-maps arrays and derives legal sequence starts from
`path_lengths`:

```text
valid starts for episode e =
    offset_e ... offset_e + length_e - sequence_length
```

This prevents a sampled sequence from joining the end of one episode to the
start of another. The reader additionally validates agent counts; array shapes
for observations, actions, rewards, legal actions, terminals, truncations, and
`filled`; the sum of `path_lengths`; and the validity of `filled.npy`.

The replay buffer returns batch-major arrays with shape `(B, T, N, ...)`.
`_prepare_training_batch()` converts them into the time-major layout
`(T, B, N, ...)` used by the losses, so `[1:]` and `[:-1]` remain adjacent
transitions from the same trajectory. Terminal and truncation indicators are
merged into a compatible done signal.

For the checked 25m Good dataset with `sequence_length=20`, the reader produced
156,316 legal sequences and excluded 94,970 flat starts that could have crossed
episode boundaries.

### 10. Preserved baseline behavior

For direct comparison with the existing two-million-step baseline, DyGFlow
preserves the following behavior:

- Q-guidance uses `argmax -> one_hot`.
- The LSTM encoder is called in the same way as in the baseline.
- The default actor and critic learning rate is `3e-4`.
- Discount, `tau`, flow steps, and hidden-layer defaults remain unchanged.
- Dataset contents and agent-ID concatenation remain unchanged.

## Validation performed

The implementation has been checked for:

- Python compilation and `git diff --check`.
- Memory-mapped sampling from the real 25m Good dataset.
- Single-group model initialization and gradient updates.
- A second gradient update after forcing a split into two groups.
- Parent-to-child copying of group hypernetworks and group embeddings.
- Centralized and decentralized online action generation.
- Legal-action masking for 25 agents.
- Saving and restoring checkpoint parameters and group identifiers.
- A complete one-step `smac_main.py` training run.
- A GPU update with `batch_size=16`, `sequence_length=20`, and 25 agents.

These checks establish that the code path runs, but they do not replace a full
performance study. Before a two-million-step run, a 100,000- to 400,000-step
pilot is recommended to inspect Q-values, gradients, group loss, group counts,
and evaluation returns.

## GPU compilation recovery

If a log reports a segmentation fault in `backend_compile` and `jax.devices()`
subsequently returns only a CPU device, restart the machine and confirm that
JAX detects a CUDA device before resuming training:

```bash
python -c "import jax; print(jax.devices())"
```

The vectorized independent hypernetwork parameters avoid expanding duplicate
Python modules for all 25 agents.

## Runtime grouping fix

An earlier implementation used GPU `Array.at` operations to warm-start group
parameters, triggering an additional PTX compilation during training. Group
identifiers were also static JIT fields, so every new assignment caused a full
training-step recompilation. The current implementation copies parent-group
parameters on the host and stores fixed-shape `group_ids` as a dynamic `int32`
JAX array. These changes affect parameter transfer and JIT caching only; they do
not alter grouping rules, copied values, loss functions, or gradients.
