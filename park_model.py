from ast import arg
from datetime import datetime
from time import time
import gym
import highway_env
from stable_baselines3 import SAC, HerReplayBuffer
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

env = gym.make("parking-v0")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='learn/run', type=str)
parser.add_argument('--steps', help='steps to learn/run', type=int)
parser.add_argument('--filename', help='name of the file if in learn mode', type=str)

args = parser.parse_args()

if args.mode is None:
    print("Please specify mode")
    exit(1)

mode = args.mode

if args.filename == None:
    print("specify filename to store/run the model")
    exit(1)

model_name = args.filename

if args.steps == None:
    print("Steps not specified running default for 10000")
    steps = 5000
else:
    steps = args.steps

if mode == 'learn':
    her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', online_sampling=True, max_episode_length=100)
    model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=her_kwargs, verbose=1, buffer_size=int(1e6),
                learning_rate=1e-3,
                gamma=0.95, batch_size=1024, tau=0.05,
                policy_kwargs=dict(net_arch=[512, 512, 512]))
    model.learn(steps)
    model.save(model_name+"_"+str(steps)+"_"+str(int(datetime.now().timestamp())))

else:
    model = SAC.load(model_name, env=env)
    for _ in range(steps):
        obs, done = env.reset(), False
        while not done:
            env.render()
            # time.sleep(1)
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, info = env.step(action)

env.terminate()

# plot training/run episode graphs
df = pd.read_csv(env.file_name_episode + ".csv")

figure = plt.gcf()
figure.set_size_inches(18, 13)

ax = figure.add_subplot(2, 2, 1)
ax.plot(df['episodes'], df['rewards'])

plt.title("Total Rewards vs Episodes", fontsize=16)
ax.set_xlabel('Episodes')
ax.set_ylabel('Total Rewards')
plt.savefig(model_name+"_"+str(steps)+"_episodes" + '.png', dpi=200, bbox_inches="tight")

# plot training/run steps graphs
df2 = pd.read_csv(env.file_name_steps + ".csv")
figure.clear(True) 
ax.cla()
ax = figure.add_subplot(2, 1, 1)
ax.plot(df2['steps'], df2['rewards'])

ax.set_title("Rewards vs Steps", fontsize=16)
ax.set_xlabel('Steps')
ax.set_ylabel('Rewards')

ax = figure.add_subplot(2, 1, 2)
ax.plot(df2['steps'], df2['velocity'])
ax.set_title("Velocity vs Steps", fontsize=16)
ax.set_xlabel('Steps')
ax.set_ylabel('Velocity')

plt.savefig(model_name+"_"+str(steps)+"_steps" + '.png', dpi=200, bbox_inches="tight")

env.close()