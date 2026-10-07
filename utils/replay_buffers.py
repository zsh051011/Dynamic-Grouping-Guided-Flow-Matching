import os
from typing import Any, Dict

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import tree
from chex import Array
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.vault import Vault

Experience = Dict[str, Array]


class FlashbaxReplayBuffer:
    def __init__(
            self,
            sequence_length: int,
            max_size: int = 50_000,
            batch_size: int = 32,
            sample_period: int = 1,
            seed: int = 42,
            store_to_vault: bool = False,
            vault_name: str = "recorded_data",
    ):
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size

        # Vault
        self._store_to_vault = store_to_vault
        self._vault_name = vault_name
        self._write_to_vault_every = max_size - 10
        self._vault_has_initialised = False
        self._step_count = 0

        # Flashbax buffer
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=sequence_length,
            period=sample_period,
            min_length_time_axis=1,
            max_size=max_size,
        )

        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add, donate_argnums=[0])

        self._buffer_state: TrajectoryBufferState = None
        self._rng_key = jax.random.PRNGKey(seed)
        self._np_rng = np.random.default_rng(seed)
        self._native_arrays = None
        self._native_valid_starts = None
        self._sample_device = None

    def add(
            self,
            observations: Dict[str, np.ndarray],
            actions: Dict[str, np.ndarray],
            rewards: Dict[str, np.ndarray],
            terminals: Dict[str, np.ndarray],
            truncations: Dict[str, np.ndarray],
            infos: Dict[str, Any],
    ) -> None:

        stacked_infos = {}
        for key, value in infos.items():
            if isinstance(value, dict):
                stacked_infos[key] = np.stack(list(value.values()), axis=0)
            else:
                stacked_infos[key] = value

        timestep = {
            "observations": np.stack(list(observations.values()), axis=0),
            "actions": np.stack(list(actions.values()), axis=0),
            "rewards": np.stack(list(rewards.values()), axis=0),
            "terminals": np.stack(list(terminals.values()), axis=0),
            "truncations": np.stack(list(truncations.values()), axis=0),
            "infos": stacked_infos,
        }

        if self._buffer_state is None:
            self._buffer_state = self._replay_buffer.init(timestep)

            # Log stuff to vault/flashbax
            if not self._vault_has_initialised and self._store_to_vault:
                self._vault = Vault(
                    vault_name=self._vault_name,
                    experience_structure=self._buffer_state.experience,
                )
                self._vault_has_initialised = True

        timestep = tree.map_structure(
            lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), timestep
        )  # add batch & time dims

        self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

        self._step_count += 1

        if self._step_count % self._write_to_vault_every == 0 and self._vault_has_initialised:
            self._vault.write(self._buffer_state)

    def sample(self) -> Experience:
        if self._native_arrays is not None:
            return self._sample_native_npys()

        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        experience = batch.experience  # type: ignore
        if self._sample_device is not None:
            experience = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self._sample_device), experience
            )
        return experience

    def populate_from_vault(
            self,
            source: str,
            env_name: str,
            scenario_name: str,
            dataset_name: str,
            rel_dir: str = "/data",
    ) -> bool:
        vault = Vault(
            vault_name=f"{source}/{env_name}/{scenario_name}.vlt",
            vault_uid=dataset_name,
            rel_dir=rel_dir,
        )

        read_on_cpu = os.environ.get("MACFLOW_VAULT_ON_CPU", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if read_on_cpu:
            with jax.default_device(jax.devices("cpu")[0]):
                self._buffer_state = vault.read()
            try:
                gpu_devices = jax.devices("gpu")
            except RuntimeError:
                gpu_devices = []
            self._sample_device = gpu_devices[0] if gpu_devices else None
        else:
            self._buffer_state = vault.read()

        # Recreate the buffer and associated pure functions
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=self._batch_size,
            sample_sequence_length=self._sequence_length,
            period=1,
            min_length_time_axis=1,
            max_size=self._sequence_length,
        )
        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)

        return True

    def populate_from_native_npys(self, dataset_dir: str, agents) -> bool:
        """Populate the replay buffer from native SMAC npy transition files.

        Expected files are generated by scripts/collect_native_smac25_from_qmix_ckpt.py:
        obs/states/actions/rewards/legals/terminals/truncations/path_lengths.
        Observations must be native SMAC observations without appended agent ids.
        """
        observations = np.load(f"{dataset_dir}/obs.npy", mmap_mode="r")
        states = np.load(f"{dataset_dir}/states.npy", mmap_mode="r")
        actions = np.load(f"{dataset_dir}/actions.npy", mmap_mode="r")
        rewards = np.load(f"{dataset_dir}/rewards.npy", mmap_mode="r")
        legals = np.load(f"{dataset_dir}/legals.npy", mmap_mode="r")
        terminals = np.load(f"{dataset_dir}/terminals.npy", mmap_mode="r")
        truncations = np.load(f"{dataset_dir}/truncations.npy", mmap_mode="r")
        filled_path = f"{dataset_dir}/filled.npy"
        filled = np.load(filled_path, mmap_mode="r") if os.path.exists(filled_path) else None
        path_lengths = np.load(f"{dataset_dir}/path_lengths.npy")

        n_steps, n_agents = actions.shape[:2]
        agents = list(agents)
        if len(agents) != n_agents:
            raise ValueError(f"agent count mismatch: env has {len(agents)}, data has {n_agents}")
        expected_agent_prefix = (n_steps, n_agents)
        if observations.shape[:2] != expected_agent_prefix:
            raise ValueError(f"obs shape {observations.shape} does not match {expected_agent_prefix}")
        if legals.shape[:2] != expected_agent_prefix:
            raise ValueError(f"legals shape {legals.shape} does not match {expected_agent_prefix}")
        checked_arrays = [
            ("actions", actions),
            ("rewards", rewards),
            ("terminals", terminals),
            ("truncations", truncations),
        ]
        if filled is not None:
            checked_arrays.append(("filled", filled))
        for name, array in checked_arrays:
            if array.shape[:2] != expected_agent_prefix:
                raise ValueError(
                    f"{name} shape {array.shape} does not match {expected_agent_prefix}"
                )
        if states.shape[0] != n_steps:
            raise ValueError("states and actions have inconsistent time dimensions")
        if filled is not None and not np.all(np.asarray(filled).astype(bool)):
            raise ValueError(
                "native dataset contains unfilled transitions; compact episodes before loading"
            )
        if int(path_lengths.sum()) != n_steps:
            raise ValueError(
                f"path_lengths sum {int(path_lengths.sum())} does not match data steps {n_steps}"
            )

        episode_offsets = np.concatenate([[0], np.cumsum(path_lengths[:-1])]).astype(np.int64)
        valid_starts = []
        for offset, length in zip(episode_offsets, path_lengths):
            n_valid = int(length) - self._sequence_length + 1
            if n_valid > 0:
                valid_starts.append(offset + np.arange(n_valid, dtype=np.int64))

        if not valid_starts:
            raise ValueError(
                f"no native SMAC episodes in {dataset_dir} are at least "
                f"sequence_length={self._sequence_length}"
            )

        self._native_valid_starts = np.concatenate(valid_starts)
        self._native_arrays = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
            "filled": filled,
            "infos": {
                "legals": legals,
                "state": states,
            },
        }
        self._step_count = n_steps
        cross_boundary = n_steps - self._sequence_length + 1 - len(self._native_valid_starts)
        print(
            "[replay_buffers] Loaded native SMAC episodes "
            f"from {dataset_dir}: steps={n_steps}, episodes={len(path_lengths)}, "
            f"valid_sequences={len(self._native_valid_starts)}, "
            f"blocked_cross_episode_sequences={cross_boundary}",
            flush=True,
        )

        return True

    def _sample_native_npys(self) -> Experience:
        starts = self._np_rng.choice(self._native_valid_starts, size=self._batch_size)
        time_offsets = np.arange(self._sequence_length, dtype=np.int64)
        # Replay samples are batch-major here.  The agent's
        # _prepare_training_batch() converts (B,T,...) to (T,B,...) before the
        # TD loss applies [1:] and [:-1].
        indices = starts[:, None] + time_offsets[None, :]
        arrays = self._native_arrays
        actions = arrays["actions"][indices]
        rewards = arrays["rewards"][indices]
        if rewards.ndim == 4 and rewards.shape[-1] == 1:
            rewards = np.squeeze(rewards, axis=-1)
        if arrays["filled"] is None:
            filled = np.ones(actions.shape, dtype=np.float32)
        else:
            filled = arrays["filled"][indices]

        return {
            "observations": jnp.asarray(arrays["observations"][indices], dtype=jnp.float32),
            "actions": jnp.asarray(actions, dtype=jnp.int32),
            "rewards": jnp.asarray(rewards, dtype=jnp.float32),
            "terminals": jnp.asarray(arrays["terminals"][indices], dtype=jnp.bool_),
            "truncations": jnp.asarray(arrays["truncations"][indices], dtype=jnp.bool_),
            "filled": jnp.asarray(filled, dtype=jnp.float32),
            "infos": {
                "legals": jnp.asarray(arrays["infos"]["legals"][indices], dtype=jnp.float32),
                "state": jnp.asarray(arrays["infos"]["state"][indices], dtype=jnp.float32),
            },
        }

    def populate_from_native_mamujoco_npys(self, dataset_dir: str, agents) -> bool:
        """Populate the replay buffer from native MA-MuJoCo npy transition files.

        Expected files are generated by scripts/collect_native_mamujoco.py:
        obs/states/actions/rewards/terminals/truncations/path_lengths.
        Observations are raw local observations without appended agent ids.
        Actions are continuous arrays.
        """
        observations = np.load(f"{dataset_dir}/obs.npy", mmap_mode="r")
        states = np.load(f"{dataset_dir}/states.npy", mmap_mode="r")
        actions = np.load(f"{dataset_dir}/actions.npy", mmap_mode="r")
        rewards = np.load(f"{dataset_dir}/rewards.npy", mmap_mode="r")
        terminals = np.load(f"{dataset_dir}/terminals.npy", mmap_mode="r")
        truncations = np.load(f"{dataset_dir}/truncations.npy", mmap_mode="r")

        n_steps, n_agents = actions.shape[:2]
        agents = list(agents)
        if len(agents) != n_agents:
            raise ValueError(f"agent count mismatch: env has {len(agents)}, data has {n_agents}")
        if observations.shape[:2] != (n_steps, n_agents):
            raise ValueError("native MA-MuJoCo obs/actions have inconsistent dimensions")
        if states.shape[0] != n_steps or rewards.shape[:2] != (n_steps, n_agents):
            raise ValueError("native MA-MuJoCo npy files have inconsistent time dimensions")
        if n_steps > self._max_size:
            raise ValueError(
                f"native MA-MuJoCo dataset has {n_steps} steps, but buffer max_size is "
                f"{self._max_size}; increase --buffer_size"
            )

        timestep = {
            "observations": np.asarray(observations[0], dtype="float32"),
            "actions": np.asarray(actions[0], dtype="float32"),
            "rewards": np.asarray(rewards[0], dtype="float32"),
            "terminals": np.asarray(terminals[0], dtype="bool"),
            "truncations": np.asarray(truncations[0], dtype="bool"),
            "infos": {"state": np.asarray(states[0], dtype="float32")},
        }

        self._buffer_state = self._replay_buffer.init(timestep)
        batch = {
            "observations": np.asarray(observations, dtype="float32")[None, ...],
            "actions": np.asarray(actions, dtype="float32")[None, ...],
            "rewards": np.asarray(rewards, dtype="float32")[None, ...],
            "terminals": np.asarray(terminals, dtype="bool")[None, ...],
            "truncations": np.asarray(truncations, dtype="bool")[None, ...],
            "infos": {"state": np.asarray(states, dtype="float32")[None, ...]},
        }
        self._buffer_state = self._buffer_add_fn(self._buffer_state, batch)
        self._step_count = n_steps

        return True
