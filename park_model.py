from ast import arg
from datetime import datetime
from time import time
from xmlrpc.client import boolean
import gym
import highway_env
from stable_baselines3 import SAC, HerReplayBuffer
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True


def run_and_save_model(steps, model_name, env, useHER = True):
    her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', online_sampling=True, max_episode_length=100)
    # if we want to use HER for training
    if useHER:
        print("running with HER")
        model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=her_kwargs, verbose=1, buffer_size=int(1e6),
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    policy_kwargs=dict(net_arch=[512, 512, 512]))
    else:
        print("running without HER")
        model = SAC('MultiInputPolicy', env,verbose=1, buffer_size=int(1e6),
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    policy_kwargs=dict(net_arch=[512, 512, 512]))
    model.learn(steps)
    model.save(model_name+"_"+str(steps)+"_"+str(int(datetime.now().timestamp())))

def run_simulation(model_name, env):
    model = SAC.load(model_name, env=env)
    for _ in range(steps):
        obs, done = env.reset(), False
        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            obs, _, done, _ = env.step(action)

def plot_graphs():
    # plot training/run episode graphs
    df = pd.read_csv(env.file_name_episode + ".csv")

    figure = plt.gcf()
    figure.set_size_inches(18, 13)

    ax = figure.add_subplot(2, 2, 1)
    # parking success/crash/elapsed points
    df_elapsed = df[df['reason'] == 'ELAPSED']
    df_carshed = df[df['reason'] == 'CRASHED']
    df_success = df[df['reason'] == 'SUCCESS']
    ax.plot(df_elapsed['episodes'], df_elapsed['rewards'], marker='o',markerfacecolor="orange",linestyle = 'None',markersize=8)
    ax.plot(df_carshed['episodes'], df_carshed['rewards'], marker='o',markerfacecolor="red",linestyle = 'None',markersize=8)
    ax.plot(df_success['episodes'], df_success['rewards'], marker='o',markerfacecolor="green",linestyle = 'None',markersize=8)

    ax.plot(df['episodes'], df['rewards'],color='blue')

    plt.legend(["Elapsed", "Crashed", "Success"])

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

    df2_new_episode = df2[df2['new_episode'] == True]

    ax.plot(df2_new_episode['steps'],df2_new_episode['rewards'], marker='o',markerfacecolor="orange",linestyle = 'None',markersize=8)

    plt.legend(["Rewards", "New Episode"])
    ax.set_title("Rewards vs Steps", fontsize=16)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Rewards')

    ax = figure.add_subplot(2, 1, 2)
    ax.plot(df2['steps'], df2['velocity'])

    ax.plot(df2_new_episode['steps'],df2_new_episode['velocity'], marker='o',markerfacecolor="orange",linestyle = 'None',markersize=8)
    ax.set_title("Velocity vs Steps", fontsize=16)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Velocity')

    plt.legend(["Velocity", "New Episode"])
    plt.savefig(model_name+"_"+str(steps)+"_steps" + '.png', dpi=200, bbox_inches="tight")


env = gym.make("parking-v0")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='learn/run', type=str)
parser.add_argument('--steps', help='steps/episodes to learn/run', type=int)
parser.add_argument('--filename', help='name of the file if in learn mode', type=str)
parser.add_argument('--her', help='Use HER',default=1, type=int)

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

useHER = args.her

if mode == 'learn':
    env.update_config({"totalSteps": steps})
    run_and_save_model(steps, model_name,env,useHER)

elif mode == 'phasedLearn':
    env.update_config({"phasedLearning": True, "totalSteps": steps})
    run_and_save_model(steps, model_name,env,useHER)

elif mode == 'phasedRun':
    env.update_config({"phasedLearning": True, "totalSteps": steps})
    run_simulation(model_name, env)
else:
    env.update_config({"totalSteps": steps})
    run_simulation(model_name, env)

env.terminate()

plot_graphs()

env.close()