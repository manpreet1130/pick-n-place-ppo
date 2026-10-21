import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_env import RobotEnvironment

LOG_DIR = "runs/ppo_pick_place"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{LOG_DIR}/best/best_model.zip")
    # EvalCallback doesn't save normalization stats with the best model, so use the latest ones
    parser.add_argument("--vecnormalize", default=f"{LOG_DIR}/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = DummyVecEnv([lambda: RobotEnvironment(render_mode=render_mode)])
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = False
    env.norm_reward = False

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
