import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from robot_env import RobotEnvironment
from robot_env_visual import RobotEnvironmentVisual

N_ENVS = 8
TOTAL_STEPS = 2_000_000

# env class, policy, which obs keys VecNormalize should touch (None = whole Box obs), entropy coef
ENVS = {
    "state": (RobotEnvironment, "MlpPolicy", None, 0.0),
    # never normalize the image; entropy bonus keeps exploration alive (last visual run's std collapsed)
    "visual": (RobotEnvironmentVisual, "MultiInputPolicy", ["proprio"], 0.01),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVS.keys(), default="state")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    env_cls, policy, norm_obs_keys, ent_coef = ENVS[args.env]
    LOG_DIR = f"runs/ppo_pick_place_{args.env}"
    os.makedirs(LOG_DIR, exist_ok=True)

    # each env runs in its own process -> its own PyBullet instance
    env = make_vec_env(env_cls, n_envs=N_ENVS, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, norm_obs_keys=norm_obs_keys)

    eval_env = make_vec_env(env_cls, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, norm_obs_keys=norm_obs_keys)
    if args.env == "visual":
        # PPO wraps the training env as VecTransposeImage(VecNormalize) itself; mirror that for eval
        eval_env = VecTransposeImage(eval_env)

    model = PPO(
        policy,
        env,
        n_steps=1024,          # per env -> 8192 steps per update
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(save_freq=100_000 // N_ENVS, save_path=f"{LOG_DIR}/checkpoints",
                           name_prefix="ppo", save_vecnormalize=True),
        # 5 episodes keeps eval cheap: each visual episode is ~400 steps on a single env
        EvalCallback(eval_env, eval_freq=50_000 // N_ENVS, n_eval_episodes=5,
                     best_model_save_path=f"{LOG_DIR}/best", deterministic=True),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks)

    model.save(f"{LOG_DIR}/final_model")
    env.save(f"{LOG_DIR}/vecnormalize.pkl")   # needed to run the policy later
    env.close()
