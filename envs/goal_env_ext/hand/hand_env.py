# Created by Xingyu Lin, 04/09/2018                                                                                  
import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from envs.goal_env_ext.goal_env_ext import GoalEnvExt


class HandEnv(GoalEnvExt):
    def __init__(self, model_path, n_substeps, initial_qpos, relative_control,
                 distance_threshold,
                 distance_threshold_obs,
                 horizon,
                 image_size,
                 reward_type='sparse',
                 use_true_reward=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 with_goal=True,
                 **kwargs):
        self.relative_control = relative_control

        super(HandEnv, self).__init__(model_path=model_path, n_substeps=n_substeps, n_actions=20,
                                      initial_qpos=initial_qpos, use_image_goal=use_image_goal,
                                      use_visual_observation=use_visual_observation,
                                      use_true_reward=use_true_reward, reward_type=reward_type,
                                      distance_threshold=distance_threshold,
                                      distance_threshold_obs=distance_threshold_obs, horizon=horizon,
                                      image_size=image_size, with_goal=with_goal, **kwargs)

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (20,)

        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        if self.relative_control:
            actuation_center = np.zeros_like(action)
            for i in range(self.sim.data.ctrl.shape[0]):
                actuation_center[i] = self.sim.data.get_joint_qpos(
                    self.sim.model.actuator_names[i].replace(':A_', ':'))
            for joint_name in ['FF', 'MF', 'RF', 'LF']:
                act_idx = self.sim.model.actuator_name2id(
                    'robot0:A_{}J1'.format(joint_name))
                actuation_center[act_idx] += self.sim.data.get_joint_qpos(
                    'robot0:{}J0'.format(joint_name))
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:palm')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 55.
        self.viewer.cam.elevation = -25.
