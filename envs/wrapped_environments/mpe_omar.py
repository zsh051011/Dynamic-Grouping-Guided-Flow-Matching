from typing import Any, Dict

import numpy as np
import re
from envs.custom_environments.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)
from envs.custom_environments.multiagent_particle_envs.multiagent.simple_spread import Scenario

from .base import BaseEnvironment, ResetReturn, StepReturn


class MPEOMAR(BaseEnvironment):
    """OMAR-style positive-reward simple spread.

    Supports scenario names such as simple_spread, simple_spread6, and
    simple_spread_6. Actions are discrete MPE movement ids: 0..4.
    """

    def __init__(self, scenario, seed=None):
        num_agents = self._parse_num_agents(scenario)
        if seed is not None:
            np.random.seed(seed)

        scenario_impl = Scenario()
        world = scenario_impl.make_world(num_agents=num_agents)

        env = MultiAgentEnv(
            world,
            scenario_impl.reset_world,
            scenario_impl.reward,
            scenario_impl.observation,
            discrete_action=True,
        )
        env.discrete_action_input = True
        self.environment = env

        self.agents = [f"agent_{n}" for n in range(num_agents)]
        self.num_actions = 5
        self.num_agents = len(self.agents)
        self.max_episode_length = 25
        self.t = 0

    @staticmethod
    def _parse_num_agents(scenario):
        match = re.search(r"spread_?(\d+)$", str(scenario))
        return int(match.group(1)) if match else 3

    def reset(self) -> ResetReturn:
        obs = self.environment.reset()

        observations = {agent: obs[i].astype("float32") for i, agent in enumerate(self.agents)}

        self.t = 0

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mpe_actions = []
        for agent in self.agents:
            mpe_actions.append(self._to_underlying_action(actions[agent]))

        next_observation, reward, done, info = self.environment.step(mpe_actions)

        terminals = {agent: done[i] for i, agent in enumerate(self.agents)}
        trunctations = {agent: False for i, agent in enumerate(self.agents)}

        rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}

        observations = {
            agent: next_observation[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        if self.t >= self.max_episode_length - 1:
            terminals = {agent: True for i, agent in enumerate(self.agents)}

        self.t += 1

        return observations, rewards, terminals, trunctations, info  # type: ignore

    @staticmethod
    def _to_underlying_action(action):
        # Dataset/checkpoint actions follow on-policy one-hot semantics:
        # 1=+x, 2=-x, 3=+y, 4=-y. The underlying OMAR MPE integer API uses
        # 1=-x, 2=+x, 3=-y, 4=+y, so convert before env.step.
        action_id = int(np.asarray(action).reshape(-1)[0])
        return {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}[action_id]

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
