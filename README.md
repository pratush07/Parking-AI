# Autonomous Parking using Reinforcement Learning

This repo contains the code for the dissertation titled "Autonomous Parking using Reinforcement Learning".

## Folder Structure
```bash
├── highway_env
├── saved_models
├── scenarios
├── tests
├── requirements.txt
├── park_test.py
├── park_model.py
├──.gitignore  
├── Readme.md
└── setup.py
```

## Installation
We have used a modified ```highway-env``` as our testing environment. To install the library open up a terminal and type the following commands:
```
cd Parking-AI
pip install -e .
```
Next we will install stable baseline (https://stable-baselines3.readthedocs.io/en/master/) directly from the github as shown below.
```
pip install git+https://github.com/DLR-RM/stable-baselines3
```
Install the remaining python dependencies by using the command below.
```
pip install -r requirements.txt
```

## Training
We will use the file park_model.py to perform our training.

1. To perform the training for a vertical slots for goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal. After the training ends, we will have a model faved by the name sac_straight_steps_timestamp.

```
python park_model.py --mode learn --episodes 100 --filename sac_straight --goalSpotNumber 2
```

2. To perform the training for a diagonal slots for goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal. After the training ends, we will have a model faved by the name sac_diag_steps_timestamp.

```
python park_model.py --mode learn --episodes 100 --filename sac_diag --goalSpotNumber 2 --diagonalShift 6
```

## Simulation
1. To perform a simulation in an vertical slots environment using a trained agent, use the command below.
```
python park_model.py --mode run --episodes 30 --filename sac_straight --goalSpotNumber 2
```
![vertical-slot](https://github.com/pratush07/Parking-AI/blob/1270c035f0bbd639924e62e904a549767aea4274/scenarios/gifs/diagonal.gif)


2. To perform a simulation in an vertical slots environment using a trained agent, use the command below.
```
python park_model.py --mode run --episodes 30 --filename sac_straight --goalSpotNumber 2 --diagonalShift 6
```
