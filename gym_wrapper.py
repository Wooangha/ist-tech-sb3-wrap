from abc import ABC, abstractmethod
from typing import Optional, Callable, TypeVar, Generic

import gym
from gym import spaces

import numpy as np

import libimmortal
from libimmortal.env import ImmortalSufferingEnv

modelInputType = TypeVar("modelInputType")
modelOutputType = TypeVar("modelOutputType")


class ObsBuilder(ABC, Generic[modelInputType]):
    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, obs: dict[str, np.ndarray]) -> modelInputType:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> modelInputType:
        raise NotImplementedError()


class ActionParser(ABC, Generic[modelOutputType]):
    @abstractmethod
    def __call__(self, model_output: modelOutputType) -> np.ndarray:
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        raise NotImplementedError()


class RewardFn(ABC):

    @abstractmethod
    def reset(self, obs: dict[str, np.ndarray]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> float:
        raise NotImplementedError()


class DoneCondition(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> bool:
        raise NotImplementedError()


class TruncateCondition(ABC):
    @abstractmethod
    def __call__(self, step_count: int) -> bool:
        raise NotImplementedError()


class GymWrapper(gym.Env, Generic[modelInputType, modelOutputType]):
    def __init__(
        self,
        env: ImmortalSufferingEnv,
        obs_builder: ObsBuilder[modelInputType],
        reward_fn: RewardFn,
        done_condition: DoneCondition,
        action_parser: ActionParser[modelOutputType],
        truncate_condition: Optional[TruncateCondition] = None,
    ):
        self.env = env
        self._obs_builder = obs_builder
        self._reward_fn = reward_fn
        self._done_condition = done_condition
        self._action_parser = action_parser
        self._truncate_condition = truncate_condition

    def reset(self) -> modelInputType:
        raw_obs = self.env.reset()

        self._reward_fn.reset(raw_obs)
        self._done_condition.reset()

        return self._obs_builder.reset(raw_obs)

    def step(self, action: modelOutputType) -> tuple[modelInputType, float, bool, dict]:
        parsed_action = self._action_parser(action)
        raw_obs, reward, done, info = self.env.step(parsed_action)

        obs = self._obs_builder(raw_obs)
        reward = self._reward_fn(raw_obs)
        done = done or self._done_condition(raw_obs)

        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()

    @property
    def observation_space(self) -> spaces.Space:
        return self._obs_builder.observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_parser.action_space
