import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage

from robot_env import RobotEnvironment
from robot_env_visual import RobotEnvironmentVisual

ENVS = {
    "state": RobotEnvironment,
    "visual": RobotEnvironmentVisual,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVS.keys(), default="state")
    # defaults to the run folder for --env (set below)
    parser.add_argument("--model", default=None)
    # EvalCallback doesn't save normalization stats with the best model, so use the latest ones
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    env_cls = ENVS[args.env]
    LOG_DIR = f"runs/ppo_pick_place_{args.env}"
    args.model = args.model or f"{LOG_DIR}/best/best_model.zip"
    args.vecnormalize = args.vecnormalize or f"{LOG_DIR}/vecnormalize.pkl"

    render_mode = None if args.no_render else "human"
    env = DummyVecEnv([lambda: env_cls(render_mode=render_mode)])
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = False
    env.norm_reward = False
    if args.env == "visual":
        # same wrapper order as training: VecTransposeImage(VecNormalize)
        env = VecTransposeImage(env)

    model = PPO.load(args.model, env=env)

    obs = env.reset()
    episode, steps, total_reward = 0, 0, 0.0
    while episode < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward[0]

        if done[0]:
            episode += 1
            print(f"episode {episode}: success={info[0]['is_success']}  steps={steps}  reward={total_reward:.1f}")
            steps, total_reward = 0, 0.0

    env.close()
