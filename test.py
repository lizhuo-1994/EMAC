import gym
import mujoco_py
env = gym.make('Humanoid-v3')

max_s = 0
min_s = 0

while True:
    current_max_s = max(env.observation_space.sample())
    current_min_s = min(env.observation_space.sample())
    if max_s < current_max_s:
        max_s = current_max_s
    if min_s > current_min_s:
        min_s = current_min_s
    print(max_s, min_s)