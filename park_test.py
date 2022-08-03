import gym
import highway_env
import time
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

env = gym.make("parking-v0")
# diagonal run
# env.update_config({"gridSizeX": 4,"goalSpotNumber": 2, "diagonalShift": 6,"duration": 150})

# vertical run
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "duration": 150})

# phased 2 scenarios
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6, "phasedLearning":True, "duration": 150,})

# random 2 scenarios
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6,"randomLearning":True, "duration": 150})

# parallel run
# env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "duration": 150, "is_parallel_parking": True})

# random 3 scenarios
env.update_config({"gridSizeX": 6,"goalSpotNumber": 2, "diagonalShift": 6, "is_parallel_parking": True,"randomLearning":True, "duration": 150})


# print(env.config)

observation = env.reset()
# img = env.render(mode="rgb_array")
# plt.imsave('scenarios/scenario45_start.png', img)

model = SAC.load("sac_random_37500_1658884549", env=env)

# for _ in range(5000):
#     # env.render()
#     img = env.render(mode="rgb_array")

#     # time.sleep(4)
#     # print(action)
#     action, _ = model.predict(observation, deterministic=True)
#     observation, reward, done, info = env.step(action) # take a random action
#     # print(observation)
#     # print(info)
#     if done:
#         break

for _ in range(20):
    obs, done = env.reset(), False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

# plt.imsave('scenarios/scenario3.png', img)
env.terminate()
env.close()