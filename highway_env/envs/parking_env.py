from abc import abstractmethod
from datetime import datetime
from gym import Env
from gym.envs.registration import register
import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark, Obstacle
import math
import csv

class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    episode_ctr = -1
    total_reward = 0
    file_name_episode = "learning_episode_stats"
    file_open_episode = None
    file_writer_episode = None

    steps_ctr = 0
    file_name_steps = "learning_steps_stats"
    file_open_steps = None
    file_writer_steps = None

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -4,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 400,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            
            #user defined configs from here
            "otherParkedVehicles" : 1, # all slots will be parked with cars but one
            "obstacleBox": 1, # obstacle box around the parking lot
            "gridSizeX": 6,# number of slots available on both sides
            "corridorWidth": 10, # manaevouring empty area in between 2 rows
            "gridSpotWidth": 4.0, # slot width  
            "gridSpotLength": 8, # length of slot
            "initialEgoPosition": [10,0], # if none will be set to [0,0].
            "initialEgoHeading": 1.5, # vehicle heading. if None will be set to random, otherwise will be 2 * pi * initialHeading
            "goalSpotNumber": 0,  # fixing goal spot. None means random.
            "laneAngle": 20 # 90 degrees means vertical.
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        if self.episode_ctr == 0:
            # open episode files
            self.file_open_episode = open(self.file_name_episode + ".csv", 'w')
            self.file_writer_episode = csv.writer(self.file_open_episode)
            self.file_writer_episode.writerow(["episodes", "rewards", "success", "reason"])

            #open steps files
            self.file_open_steps = open(self.file_name_steps + ".csv", 'w')
            self.file_writer_steps = csv.writer(self.file_open_steps)
            self.file_writer_steps.writerow(["steps", "rewards", "velocity"])

        self._create_road()
        self._create_vehicles()

        self.episode_ctr += 1

    def _create_road(self) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        spots = self.config["gridSizeX"]
        width = self.config["gridSpotWidth"]
        length = self.config["gridSpotLength"]
        
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = self.config["corridorWidth"]

        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            # straight lane
            if self.config['laneAngle'] == 90:
                net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
                net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))
            
            else:
            # lane at angle
                net.add_lane("a", "b", StraightLane([x, y_offset], [x-3, y_offset+length], width=width, line_types=lt, align_lane_marking=True))
                net.add_lane("b", "c", StraightLane([x+3, -y_offset], [x, -y_offset-length], width=width, line_types=lt, align_lane_marking=True))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []

        # default position and heading of ego vehicle
        vehicle_position = [0, 0]
        vehicle_heading = 2*np.pi*self.np_random.rand()
        
        if self.config["initialEgoPosition"]:
            vehicle_position = self.config["initialEgoPosition"]
        if self.config["initialEgoHeading"]:
            vehicle_heading = 2*np.pi*self.config["initialEgoHeading"]

        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, vehicle_position, vehicle_heading, 0)            
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        if self.config["goalSpotNumber"] != None:
            lane = self.road.network.lanes_list()[self.config["goalSpotNumber"]]
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)

        # create dummy vehicles only if flag is on
        if self.config["otherParkedVehicles"]:
            self._create_dummy_vehicles(lane)
        
        # if obstacle flag is on
        if self.config["obstacleBox"]:
            self._create_obstacles()
    
    def _create_dummy_vehicles(self, goal_spot: StraightLane) -> None:
        length = self.config["gridSpotLength"]
        self.dummy_vehicles = []
        for spot in self.road.network.lanes_list():
            if not goal_spot == spot:
                vehicle = self.action_type.vehicle_class(self.road, spot.position(length/2,0), spot.heading, 0)
                self.road.vehicles.append(vehicle)
                self.dummy_vehicles.append(vehicle)

    def _create_obstacles(self) -> None:
        #Create row of obstacles at the bottom of parking grid
        extendOffsetX = 3
        extendOffsetY = 6

        xGridBottomRowObstacleOffset = -(self.config["gridSizeX"]/2 + extendOffsetX) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xGridBottomRowObstacleOffset -= self.config["gridSpotWidth"]/2

        yBottomRowObstaclePosition = self.config["gridSpotLength"] + self.config["corridorWidth"]/2 + extendOffsetY

        for i in range(self.config["gridSizeX"] + (extendOffsetX*2)):
            obstacle = Obstacle(self.road, [i*self.config["gridSpotWidth"] + xGridBottomRowObstacleOffset, yBottomRowObstaclePosition], 0, 0)
            self.road.objects.append(obstacle)


        #Create row of obstacles at the top of parking grid
        xGridTopRowObstacleOffset = -(self.config["gridSizeX"]/2 + extendOffsetX) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xGridTopRowObstacleOffset -= self.config["gridSpotWidth"]/2

        yTopRowObstaclePosition = - (self.config["gridSpotLength"] + self.config["corridorWidth"]/2 + extendOffsetY)

        for i in range(self.config["gridSizeX"] + (extendOffsetX * 2)):
            obstacle = Obstacle(self.road, [i*self.config["gridSpotWidth"] + xGridTopRowObstacleOffset, yTopRowObstaclePosition], math.pi, 0)
            self.road.objects.append(obstacle)


    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        # add check for crash condition
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        result = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

        if(isinstance(result, np.ndarray)):
            return result

        # rewarding for speed below 3
        # if abs(self.vehicle.speed) < 4:
        #     result += 0.1
        # else:
        #     # penalize for high speed
        #     result -= 0.2
        
        # if the vehicle might collide in trajectory time steps ahead, penalize
        if self.config["otherParkedVehicles"]:
            ego_vehicle = self.controlled_vehicles[0]
            for vehicle in self.dummy_vehicles:
                _,will_collide,_ = ego_vehicle._is_colliding(vehicle,10000)
                if will_collide:
                    print("vehicle might collide with " + str(vehicle))
                    result += -0.3
                # print("distance from ego to " + str(vehicle))
                # print(ego_vehicle.lane_distance_to(vehicle))

        if crashed:
            print("### crash occured" + str(result + self.config["collision_reward"]))
            return result + self.config["collision_reward"]
        
        # elif result > -self.config["success_goal_reward"]:
        #     print("### vehicle parked " + str(result + 10))
        #     return result + 10
        
        # print( "***" + str(result))
        # print("speed was " + str(self.vehicle.speed))

        self.file_writer_steps.writerow([self.steps_ctr, result, self.vehicle.speed])
        self.steps_ctr += 1

        return result

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        self.total_reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
                     for agent_obs in obs)
        return self.total_reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)

        # write to csv only if it is terminal
        if time or crashed or success:
            reason = "NA"
            if crashed:
                reason = "CRASHED"
            elif time:
                reason = "ELAPSED"
            elif success:
                reason = "SUCCESS"
            
            self.file_writer_episode.writerow([self.episode_ctr, self.total_reward, success, reason])

        return time or crashed or success

    def terminate(self):
        self.file_open_episode.close()
        self.file_open_steps.close()
        print("last step " + str(self.steps_ctr))


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


register(
    id='parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
)

register(
    id='parking-ActionRepeat-v0',
    entry_point='highway_env.envs:ParkingEnvActionRepeat'
)

