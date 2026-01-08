import os
import argparse
import numpy as np

import gymnasium as gym
# from gymnasium import spaces
# import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env


from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from libimmortal import ImmortalSufferingEnv
from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation, find_free_tcp_port

import gym_wrapper

# -------------------------------
# 3) 학습 실행부
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    # libimmortal 인자
    parser.add_argument(
        "--game_path",
        type=str,
        default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64",
        help="Immortal Suffering 실행 파일 경로",
    )
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--time_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--verbose", action="store_true")

    # 학습/환경 인자
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4, help="병렬 환경 개수")
    parser.add_argument("--episode_max_steps", type=int, default=2000, help="에피소드 Truncate 스텝")
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="auto")  # "cuda", "cpu", "auto"

    # PPO 하이퍼파라미터(초기 안전값)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=2048, help="각 env 당 rollout 길이")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_epochs", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 병렬 환경 생성
    # 주의: 같은 포트를 여러 프로세스가 쓰면 충돌하므로, 포트를 오프셋 주는 방식 권장
    # 여기서는 간단히 포트 오프셋 적용
    def make_env_with_port(rank):
        def _thunk():
            def env_builder():
                return ImmortalSufferingEnv(
                    game_path=args.game_path,
                    port=find_free_tcp_port(),
                    time_scale=args.time_scale,
                    seed=args.seed + rank,
                    width=args.width,
                    height=args.height,
                    verbose=args.verbose,
                )
            import default_settings
            return gym_wrapper.GymWrapper(
                env_builder=env_builder,
                action_parser=default_settings.DefaultActionParser(spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2])),
                obs_builder=default_settings.DefaultObsBuilder(spaces.Box(low=-np.inf, high=np.inf, shape=(103,), dtype=np.float32)),
                reward_fn=default_settings.DefaultRewardFn(),
                done_condition=default_settings.DefaultDoneCondition(args.episode_max_steps),
            )

        return _thunk

    if args.n_envs > 1:
        env = SubprocVecEnv([make_env_with_port(i) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env_with_port(0)])

    print(env.observation_space)
    print(type(env.reset()), env.reset().shape)

    # 체크포인트 콜백
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),  # 벡터환경 스텝 기준
        save_path=args.save_dir,
        name_prefix="ppo_libimmortal",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # PPO 모델 생성 (Dict obs → MultiInputPolicy)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,  # per-env rollout length
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        device=args.device,
        tensorboard_log=args.log_dir,
        verbose=1,
    )

    # 학습
    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb])

    # 최종 저장
    final_path = os.path.join(args.save_dir, "ppo_libimmortal_final")
    model.save(final_path)
    print(f"[Saved] {final_path}")

    env.close()


if __name__ == "__main__":
    main()
