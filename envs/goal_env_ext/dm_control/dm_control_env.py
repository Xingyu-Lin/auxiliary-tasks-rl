import gym
from gym import spaces
from gym.utils import seeding
import os
import numpy as np
from numpy.random import random
import time
from os import path
import cv2 as cv
from dm_control import mujoco
from dm_control.suite import base
from dm_control import mujoco, viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_control.rl.control import PhysicsError
from dm_control.mujoco import Physics
from dm_control.utils import io as resources
from dm_control.suite.utils import randomizers
from dm_control.utils import xml_tools
from envs.goal_env_ext.goal_env_ext import GoalEnvExt
from termcolor import colored
from lxml import etree
from dm_control.utils import io as resources
from envs.goal_env_ext.dm_control import common


class DmControlEnv(GoalEnvExt):
    def __init__(self, model_path, n_substeps, initial_qpos,
                 distance_threshold,
                 distance_threshold_obs,
                 horizon,
                 image_size,
                 n_actions,
                 reward_type='sparse',
                 use_image_goal=False,
                 with_goal=True,
                 use_visual_observation=True,
                 default_camera_name='fixed',
                 fully_observable=True,
                 **kwargs):

        """
        :param model_path:
        :param distance_threshold:
        :param frame_skip:
        :param horizon:
        :param goal_range:
        :param image_size:
        """
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.model_path = fullpath
        model_string, assets = self.get_model_and_assets()
        self.physics = Physics.from_xml_string(model_string,
                                               assets=assets)  # export MUJOCO_GL=osmesa to resolve mujoco context error
        # self.physics = Physics.from_xml_string(*self.get_model_and_assets1(fullpath))
        self._init_configure()
        self.time_step = 0
        self.random = np.random.RandomState(None)
        self._fully_observable = fully_observable
        super(DmControlEnv, self).__init__(model_path=self.model_path, n_substeps=n_substeps, n_actions=n_actions,
                                           initial_qpos=initial_qpos, use_image_goal=use_image_goal,
                                           use_visual_observation=use_visual_observation, reward_type=reward_type,
                                           distance_threshold=distance_threshold,
                                           distance_threshold_obs=distance_threshold_obs, horizon=horizon,
                                           image_size=image_size, with_goal=with_goal, dm_control_env=True,
                                           default_camera_name=default_camera_name, **kwargs)

    def _set_action(self, action):
        try:
            self.physics.set_control(action.continuous_actions)
        except AttributeError:
            self.physics.set_control(action)

        try:
            for _ in range(self.n_substeps):
                self.physics.step()
        except PhysicsError as ex:
            print(colored(ex, 'red'))

    # def step(self, action):
    #     action = action.flatten()
    #     # action = np.clip(action, self.action_space.low, self.action_space.high)
    #
    #     try:
    #         self._set_action(action)
    #     except NotImplementedError:
    #         try:
    #             self.physics.set_control(action.continuous_actions)
    #         except AttributeError:
    #             self.physics.set_control(action)
    #
    #     try:
    #         for _ in range(self.n_substeps):
    #             self.physics.step()
    #     except PhysicsError as ex:
    #         print(colored(ex, 'red'))
    #
    #     obs = self._get_obs()
    #
    #     if self.use_auxiliary_info:
    #         NotImplementedError
    #     else:
    #         aug_info = {}
    #
    #     state_info = self.get_current_info()
    #     info = {**aug_info, **state_info}
    #     reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
    #     self.time_step += 1
    #     done = False
    #
    #     if self.time_step >= self.horizon:
    #         done = True
    #     return obs, reward, done, info

    def _is_success(self, achieved_goal, desired_goal):

        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def get_model_and_assets(self):
        """Returns a tuple containing the model XML string and a dict of assets.

        Args:
          n_joints: An integer specifying the number of joints in the swimmer.

        Returns:
          A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
          `{filename: contents_string}` pairs.
        """

        return self._make_model(), common.ASSETS

    def read_model(self, model_filename):
        """Reads a model XML file and returns its contents as a string."""
        return resources.GetResource(model_filename)

    def _init_configure(self):
        pass

    def render(self, mode='rgb_array', image_size=None, camera_name=None, depth=True):
        if image_size is None:
            image_size = self.image_size

        if camera_name is None:
            camera_name = self.default_camera_name

        if not depth:
            obs = self.physics.render(height=image_size, width=image_size, camera_id=camera_name, depth=depth)
        else:
            img = self.physics.render(height=image_size, width=image_size, camera_id=camera_name, depth=False)
            depth_img = self.physics.render(height=image_size, width=image_size, camera_id=camera_name, depth=True)
            obs = np.dstack((img, depth_img))

        return obs
