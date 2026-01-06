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
        """Reset the observation builder state when the environment is reset."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> modelInputType:
        """
        Build observation from the raw environment observation.
        Args:
            obs (dict[str, np.ndarray]): The raw observation from the environment.
        Returns:
            modelInputType: The processed observation to be used by the model.
        """
        raise NotImplementedError()


class ActionParser(ABC, Generic[modelOutputType]):

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, model_output: modelOutputType) -> np.ndarray:
        """
        Parse the model output into a valid action.
        Args:
            model_output (modelOutputType): The raw output from the model.
        Returns:
            np.ndarray: The parsed action to be taken in the environment.
        """
        raise NotImplementedError()


class RewardFn(ABC):

    @abstractmethod
    def reset(self, obs: dict[str, np.ndarray]) -> None:
        """Reset the reward function state when the environment is reset."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> float:
        """
        Compute the reward based on the current observation.
        Args:
            obs (dict[str, np.ndarray]): The current observation from the environment.
        Returns:
            float: The computed reward.
        """
        raise NotImplementedError()


class DoneCondition(ABC):

    @abstractmethod
    def reset(self) -> None:
        """Reset the done condition when the environment is reset."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> bool:
        """
        Determine if the episode is done based on the current observation.
        Args:
            obs (dict[str, np.ndarray]): The current observation from the environment.
        Returns:
            bool: True if the episode is done, False otherwise.
        """
        raise NotImplementedError()


class TruncateCondition(ABC):

    @abstractmethod
    def reset(self) -> None:
        """Reset the truncate condition when the environment is reset."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> bool:
        """
        Determine if the episode should be truncated based on the step count.
        Args:
            obs (dict[str, np.ndarray]): The current observation from the environment.
        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        raise NotImplementedError()


class NewEnvCondition(ABC):

    @abstractmethod
    def __call__(self) -> bool:
        """Determine if a new environment should be created."""
        raise NotImplementedError()


def _parse_observation(obs: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "graphic": obs[0],  # Graphic observation
        "vector": obs[1],  # Vector observation
    }


class GymWrapper(gym.Env, Generic[modelInputType, modelOutputType]):
    """
    A Gym wrapper for Immortal Suffering environments.
    This wrapper allows the use of custom observation builders, action parsers,
    reward functions, and done conditions.
    Attributes:
        env (ImmortalSufferingEnv): The underlying Immortal Suffering environment.
        obs_builder (ObsBuilder[modelInputType]): The observation builder for the environment.
        action_parser (ActionParser[modelOutputType]): The action parser for the environment.
        reward_fn (RewardFn): The reward function for the environment.
        done_condition (DoneCondition): The done condition for the environment.
        truncate_condition (Optional[TruncateCondition]): The truncate condition for the environment. Not implemented in this wrapper.
        new_env_condition (Optional[NewEnvCondition]): The condition to create a new environment.

    Methods:
        reset() -> modelInputType: Resets the environment and returns the initial observation.
        step(action: modelOutputType) -> tuple[modelInputType, float, bool, dict]: Takes a step in the environment using the provided action.
        close() -> None: Closes the environment.
        observation_space -> spaces.Space: The observation space of the environment.
        action_space -> spaces.Space: The action space of the environment.

    """
    def __init__(
        self,
        env_builder: Callable[[], ImmortalSufferingEnv],
        obs_builder: ObsBuilder[modelInputType],
        action_parser: ActionParser[modelOutputType],
        reward_fn: RewardFn,
        done_condition: DoneCondition,
        truncate_condition: Optional[TruncateCondition] = None,
    ):
        self._env_builder = env_builder
        self._obs_builder = obs_builder
        self._reward_fn = reward_fn
        self._done_condition = done_condition
        self._action_parser = action_parser
        self._truncate_condition = truncate_condition
        self.env = self._env_builder()

    def reset(self) -> modelInputType:
        self.env.close()
        self.env = self._env_builder()

        raw_obs = self.env.reset()
        raw_obs = _parse_observation(raw_obs)
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
