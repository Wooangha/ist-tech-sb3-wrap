import gym
from gym import spaces

import numpy as np

from gym_wrapper import modelInputType, modelOutputType, ActionParser, ObsBuilder, RewardFn, DoneCondition, GymWrapper


class DefaultActionParser(ActionParser[np.ndarray]):
    def __init__(self, action_space: spaces.Space):
        self._action_space = action_space

    def __call__(self, model_output: np.ndarray) -> np.ndarray:
        return model_output

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space


class DefaultObsBuilder(ObsBuilder[np.ndarray]):
    def __init__(self, observation_space: spaces.Space):
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_space.shape, dtype=np.float32)

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return obs["vector"]

    def reset(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return obs["vector"]

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space


class DefaultDoneCondition(DoneCondition):
    def __init__(self, max_steps: int):
        self._max_steps = max_steps
        self._current_steps = 0

    def __call__(self, obs: dict[str, np.ndarray]) -> bool:
        self._current_steps += 1
        return self._current_steps >= self._max_steps

    def reset(self) -> None:
        self._current_steps = 0


class DefaultRewardFn(RewardFn):
    def reset(self, obs: dict[str, np.ndarray]) -> None:
        pass

    def __call__(self, obs: dict[str, np.ndarray]) -> float:
        return 0.0
