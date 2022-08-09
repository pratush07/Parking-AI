import gym
import highway_env
import time
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from scripts.utils import record_videos

env = gym.make("parking-v0")
# vertical run
env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "duration": 150})

# diagonal run
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6,"duration": 150})

# phased 2 scenarios
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6, "phasedLearning":True, "duration": 150,})

# random 2 scenarios
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6,"randomLearning":True, "duration": 150})

# parallel run
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "duration": 150, "is_parallel_parking": True})

# random 3 scenarios
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6, "is_parallel_parking": True,"randomLearning":True, "duration": 150})

observation = env.reset()
model = SAC.load("saved_models/sac_random_3envs_180000_1659645070", env=env)

# for _ in range(5000):
#     img = env.render(mode="rgb_array")
#     action, _ = model.predict(observation, deterministic=True)
#     observation, reward, done, info = env.step(action) # take a random action
#     print(observation)
#     print(info)
#     if done:
#         break

for _ in range(100):
    obs, done = env.reset(), False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

# plt.imsave('scenarios/scenario345.png', img)
env.terminate()
env.close()