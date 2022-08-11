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

1. To perform the training for different lane orientations to park in the  goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal. After the training ends, we will have a model saved by the name filename.

Vertical
```
python park_model.py --mode learn --episodes 100 --filename sac_straight --goalSpotNumber 2
```

Diagonal
```
python park_model.py --mode learn --episodes 100 --filename sac_diag --goalSpotNumber 2 --diagonalShift 6
```

Parallel
```
python park_model.py --mode learn --episodes 100 --filename sac_parallel --goalSpotNumber 2 --parallelParking 1
```

2. To perform the phased training for 2 lane orientations, that is vertical and diagonal, to park in goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal.
```
python park_model.py --mode phasedLearn --episodes 100 --filename sac_phased --goalSpotNumber 2 --diagonalShift 6
```

3. To perform the random training for 2 lane orientations to park in the goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal.
```
python park_model.py --mode randomLearn --episodes 100 --filename sac_random --goalSpotNumber 2 --diagonalShift 6
```

4. To perform the random training same as previous but now with parallel parking to park in the goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal.
```
python park_model.py --mode randomLearn --episodes 100 --filename sac_random --goalSpotNumber 2 --diagonalShift 6 --parallelParking 1
```

## Simulation
We will use the file park_model.py to perform our simulation.

1. To perform a simulation in different lane orientations using a trained agent, use the commands below.
Vertical
```
python park_model.py --mode run --episodes 30 --filename sac_straight --goalSpotNumber 2
```
![vertical-slot](https://github.com/pratush07/Parking-AI/blob/1270c035f0bbd639924e62e904a549767aea4274/scenarios/gifs/vertical.gif)

Diagonal
```
python park_model.py --mode run --episodes 30 --filename sac_diagonal --goalSpotNumber 2 --diagonalShift 6
```
![diagonal-slot](https://github.com/pratush07/Parking-AI/blob/1270c035f0bbd639924e62e904a549767aea4274/scenarios/gifs/diagonal.gif)

Parallel
```
python park_model.py --mode run --episodes 30 --filename sac_parallel --goalSpotNumber 2 --parallelParking 1
```
![diagonal-slot](https://github.com/pratush07/Parking-AI/blob/1270c035f0bbd639924e62e904a549767aea4274/scenarios/gifs/parallel.gif)

2. To perform the phased simulation for 2 lane orientations to park in goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal.
```
python park_model.py --mode phasedRun --episodes 100 --filename sac_phased --goalSpotNumber 2 --diagonalShift 6
```

3. To perform the random simulation for 2 lane orientations to park in the goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal.
```
python park_model.py --mode randomRun --episodes 100 --filename sac_random --goalSpotNumber 2 --diagonalShift 6
```

4. To perform the random simulation for 3 lane orientation to park in the goal slot indexed at 2 for 100 episodes, the command below can be executed in the terminal.
```
python park_model.py --mode randomRun --episodes 100 --filename sac_random --goalSpotNumber 2 --diagonalShift 6 --parallelParking 1
```

## Commandline Arguments Table
Refer this section for all the commandline arguments.
![args-table](https://github.com/pratush07/Parking-AI/blob/c6190f6c0b4061432f76cdb5b27cc217359f3a03/scenarios/args_table.png)

## References
https://github.com/eleurent/highway-env <br>
https://github.com/DinisMoreira/Dissert <br>
https://stable-baselines3.readthedocs.io/en/master/
