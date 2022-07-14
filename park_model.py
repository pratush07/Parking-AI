import gym
import highway_env
from stable_baselines3 import SAC, HerReplayBuffer
import time

env = gym.make("parking-v0")
model_name = "samplerun"
steps = 1000

her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', online_sampling=True, max_episode_length=100)
model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs, verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=1024, tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512, 512]))
model.learn(steps)
model.save(model_name)

# model = SAC.load(model_name, env=env)
# for _ in range(100):
#     obs, done = env.reset(), False
#     while not done:
#         # time.sleep(1)
#         env.render()
#         action, _ = model.predict(obs, deterministic=True)
#         print(action)
#         obs, reward, done, info = env.step(action)

# for _ in range(100):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    # print(info)
    # if done:
    #     break



# env.close()

# replicating env in gym from unity using vanilla ppo
# add parking both sides
# parallel alternative scenarios -> different parking lots single slots, and then multiple slots, 

# think about why algo performs worse in varied env, continual RL