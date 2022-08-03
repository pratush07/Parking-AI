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


def run_and_save_model(steps, model_name, env, useBuffer = 1):
    her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', online_sampling=True, max_episode_length=100)
    # if we want to use HER for training
    if useBuffer == 1:
        print("running with HER")
        model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=her_kwargs, verbose=1, buffer_size=int(1e6),
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    policy_kwargs=dict(net_arch=[512, 512, 512]))
    elif useBuffer == 2:
        print("running without HER")
        model = SAC('MultiInputPolicy', env,verbose=1, buffer_size=int(1e6),
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    policy_kwargs=dict(net_arch=[512, 512, 512]))
    elif useBuffer == 0:
        print("running with zero buffer")
        model = SAC('MultiInputPolicy', env,verbose=1, buffer_size=0,
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    policy_kwargs=dict(net_arch=[512, 512, 512]))
    model.learn(steps)
    model.save(model_name+"_"+str(steps)+"_"+str(int(datetime.now().timestamp())))

def run_simulation(episodes, model_name, env, timeDelay=None):
    model = SAC.load(model_name, env=env)
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            env.render()
            if timeDelay:
                time.sleep(timeDelay)
            action, _ = model.predict(obs, deterministic=True)
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
parser.add_argument('--episodes', help='episodes to learn/run', type=int)
parser.add_argument('--filename', help='name of the file if in learn mode', type=str)
parser.add_argument('--buffer', help='0 means no buffer. 1 means her, 2 means default buffer.',default=1, type=int)
parser.add_argument('--saveGraphs',help='save graphs',default=1, type=int)
parser.add_argument('--timeDelay',help='timeDelay during run',default=None, type=float)
parser.add_argument('--parallelParking',help='Adds parallel parking',default=0, type=int)


parser.add_argument('--gridSizeX', help='number of grid slots in each row',default=6, type=int)
parser.add_argument('--diagonalShift', help='angle of the lane',default=0, type=int)
parser.add_argument('--goalSpotNumber', help='fixed goal spot',default=1, type=int)
parser.add_argument('--duration', help='duration of each episode',default=150, type=int)

args = parser.parse_args()

if args.mode is None:
    print("Please specify mode")
    exit(1)

mode = args.mode

if args.filename == None:
    print("specify filename to store/run the model")
    exit(1)

model_name = args.filename

if args.episodes == None:
    print("Episodes not specified running default for 10")
    episodes = 10
else:
    episodes = args.episodes

useBuffer = args.buffer
gridSizeX = args.gridSizeX
diagonalShift = args.diagonalShift
goalSpotNumber = args.goalSpotNumber
duration = args.duration
saveGraphs = args.saveGraphs
timeDelay = args.timeDelay
parallelParking = args.parallelParking

# number of episodes times duration of each episode will give us the number of steps. Only needed for learning
steps = episodes * duration
print("\nTotal steps to be run is : " + str(steps))

common_env_config = {"totalEpisodes": episodes, "diagonalShift": diagonalShift, "gridSizeX": gridSizeX, "goalSpotNumber": goalSpotNumber, "duration": duration}

if parallelParking:
    common_env_config["is_parallel_parking"] = True

print("running with the following configurations..")
print("####\n" + str(common_env_config) + str("\n ####"))

if mode == 'learn':
    env.update_config(common_env_config)
    run_and_save_model(steps, model_name,env,useBuffer)

elif mode == 'phasedLearn':
    env_phased_config = common_env_config.copy()
    env_phased_config['phasedLearning'] = True
    print("Running in phased learn mode..")
    env.update_config(env_phased_config)
    run_and_save_model(steps, model_name,env,useBuffer)

elif mode == 'randomLearn':
    env_phased_config = common_env_config.copy()
    env_phased_config['randomLearning'] = True
    print("Running in random learn mode..")
    env.update_config(env_phased_config)
    run_and_save_model(steps, model_name,env,useBuffer)

elif mode == 'phasedRun':
    env_phased_config = common_env_config.copy()
    env_phased_config['phasedLearning'] = True
    print("Running in phased run mode..")
    env.update_config(env_phased_config)
    run_simulation(episodes,model_name, env, timeDelay)

elif mode == 'randomRun':
    env_phased_config = common_env_config.copy()
    env_phased_config['randomLearning'] = True
    print("Running in random run mode..")
    env.update_config(env_phased_config)
    run_simulation(episodes,model_name, env, timeDelay)

elif mode == 'run':
    env.update_config(common_env_config)
    run_simulation(episodes, model_name, env, timeDelay)
else:
    print('Mode not found!')

env.terminate()

if saveGraphs:
    print("saving graphs")
    plot_graphs()

env.close()