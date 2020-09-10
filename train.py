from datetime import datetime
import time
import pickle
import numpy as np
import torch
import gym
import argparse
import shutil
import os

from torch.utils.tensorboard import SummaryWriter

from models.utils import ReplayBuffer
from models.TD3 import TD3
from models.DDPG import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def estimate_true_q(policy, env_name, buffer, eval_episodes=1000):
    t1 = time.time()
    eval_env = gym.make(env_name)

    qs = []
    for _ in range(eval_episodes):
        eval_env.reset()

        ind = np.random.choice(buffer.size)
        state = buffer.next_state[ind]
        reward = buffer.reward[ind]

        qpos = state[:eval_env.model.nq-1]
        qvel = state[eval_env.model.nq-1:]
        qpos = np.concatenate([[0], qpos])

        eval_env.set_state(qpos, qvel)

        q = reward
        s_i = 1
        while True:
            action = policy.select_action(np.array(state))
            state, r, d, _ = eval_env.step(action)
            q += r * args.discount ** s_i

            s_i += 1

            if d:
                break
        qs.append(q)

    print("Estimation took: ", time.time() - t1)

    return np.mean(qs)


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
        parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
        parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
        parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99)                 # Discount factor
        parser.add_argument("--tau", default=0.005)                     # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
        parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
        parser.add_argument("--save_buffer", default=0)
        parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--save_model_every", type=int, default=1000000)      # Save model every timesteps
        parser.add_argument("--exp_name", default="test")
        args = parser.parse_args()

        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        dt = datetime.now()
        exp_dir = dt.strftime("%b_%d_%Y")
        exp_dir = f"./results/{exp_dir}_{args.policy}_{args.env}_{args.seed}_{args.exp_name}"
        if os.path.exists(exp_dir):
            ans = input(f"Directory {exp_dir} already exists. Overwrite? [Y/n] ")
            if ans == "Y":
                shutil.rmtree(exp_dir)
            else:
                raise Exception("Trying to rewrite existing experiment. Exiting...")
        print(f"Saving dir: {exp_dir}")

        folders = ["models", "buffers", "tb"]
        for fold in folders:
            fn = f"{exp_dir}/{fold}"
            if not os.path.exists(fn):
                os.makedirs(fn)

        env = gym.make(args.env)

        # Logger
        tb_logger = SummaryWriter(f"{exp_dir}/tb")

        # Set seeds
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
            "device": args.device,
            "log_dir": f"{exp_dir}/tb"
        }

        # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3(**kwargs)
        elif args.policy == "DDPG":
            policy = DDPG(**kwargs)

        if args.load_model != "":
            policy.load(f"{exp_dir}/models/{args.load_model}")

        replay_buffer = ReplayBuffer(state_dim, action_dim, device=args.device)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy, args.env, args.seed)]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        q_estim, q_critic = [], []

        for t in range(int(args.max_timesteps)+1):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                ep_reward = eval_policy(policy, args.env, args.seed)
                tb_logger.add_scalar("agent/eval_reward", ep_reward, t)

            # Save model
            if args.save_model and t % args.save_model_every == 0:
                print("Saving model...")
                policy.save(f"{exp_dir}/models/model_step_{t}")

            if t % 100000 == 0 and args.save_buffer:
                print(f"Saving buffer at {t} timestep...")
                replay_buffer.save(f"{exp_dir}/buffers/replay_buffer")