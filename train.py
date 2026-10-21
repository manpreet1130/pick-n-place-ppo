import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from robot_env import RobotEnvironment

LOG_DIR = "runs/ppo_pick_place"
N_ENVS = 8
TOTAL_STEPS = 2_000_000

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    # each env runs in its own process -> its own PyBullet instance
    env = make_vec_env(RobotEnvironment, n_envs=N_ENVS, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(RobotEnvironment, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,          # per env -> 8192 steps per update
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(save_freq=100_000 // N_ENVS, save_path=f"{LOG_DIR}/checkpoints",
                           name_prefix="ppo", save_vecnormalize=True),
        EvalCallback(eval_env, eval_freq=50_000 // N_ENVS, n_eval_episodes=10,
                     best_model_save_path=f"{LOG_DIR}/best", deterministic=True),
    ]

    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)

    model.save(f"{LOG_DIR}/final_model")
    env.save(f"{LOG_DIR}/vecnormalize.pkl")   # needed to run the policy later
    env.close()
