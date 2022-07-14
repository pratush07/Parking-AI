import gym
import highway_env
import time
import matplotlib.pyplot as plt

env = gym.make("parking-v0")

# print(env.config)

observation = env.reset()
print(observation)
# print(env.action_space.high)
# print(env.action_space.low)


for _ in range(1):
    # env.render()
    img = env.render(mode="rgb_array")
    plt.imsave('parking-originalview.png', img)
    action = env.action_space.sample()
    # print(action)
    # observation, reward, done, info = env.step(action) # take a random action
    # print(observation)
    # print(info)
    # if done:
    #     break

env.close()

# replicating env in gym from unity using vanilla ppo
# add parking both sides
# parallel alternative scenarios -> different parking lots single slots, and then multiple slots, 

# think about why algo performs worse in varied env, continual RL