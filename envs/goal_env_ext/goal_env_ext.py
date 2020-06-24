# Created by Xingyu Lin, 30/08/2018                                                                                  
import os
from os import path
import numpy as np
from gym import GoalEnv
from gym import spaces
from gym.utils import seeding

import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer

import cv2 as cv
import copy


def cv_render(img, name='GoalEnvExt', scale=5):
    '''Take an image in ndarray format and show it with opencv. '''
    img = img[:, :, :3]
    new_img = img[:, :, (2, 1, 0)] / 256.
    h, w = new_img.shape[:2]
    new_img = cv.resize(new_img, (w * scale, h * scale))
    cv.imshow(name, new_img)
    cv.waitKey(20)


class GoalEnvExt(GoalEnv):
    def __init__(self, model_path, n_substeps, n_actions, horizon, image_size, use_image_goal,
                 use_visual_observation, with_goal,
                 reward_type, distance_threshold, distance_threshold_obs,
                 initial_qpos=None, default_camera_name='external_camera_0', use_auxiliary_info=False, **kwargs):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "./assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model, nsubsteps=n_substeps)

        self.data = self.sim.data
        self.viewer = None
        self.np_random = None
        self.seed()

        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.horizon = horizon
        self.image_size = image_size
        self.use_image_goal = use_image_goal
        self.use_visual_observation = use_visual_observation
        self.with_goal = with_goal

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.distance_threshold_obs = distance_threshold_obs

        self._max_episode_steps = horizon
        self.time_step = 0

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.goal_state = self.goal_observation = None

        self.default_camera_name = default_camera_name
        self._set_camera()
        self.sim.render(camera_name=default_camera_name, width=self.image_size, height=self.image_size, depth=False,
                        mode='offscreen')

        self.use_auxiliary_info = use_auxiliary_info
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        obs = self.reset()

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.goal_dim = np.prod(obs['achieved_goal'].shape)
        self.goal_state_dim = np.prod(self.goal_state.shape)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal. (only one of them)
        achieved_goal = info['ag_state']
        desired_goal = info['g_state']
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])

        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # methods to override:
    # ----------------------------
    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        raise NotImplementedError

    def _get_obs(self):
        """
        Get observation
        """
        raise NotImplementedError

    def _set_action(self, ctrl):
        """
        Do simulation
        """
        raise NotImplementedError

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _set_camera(self):
        pass

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        raise NotImplementedError

    def _viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def set_hidden_goal(self):
        """
        Hide the goal position from image observation
        """
        pass

    def get_image_obs(self, depth=True, hide_overlay=True, camera_id=-1):
        assert False
        return

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        return None

    # Core functions framework
    # -----------------------------

    def reset(self):
        '''
        Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.
        '''

        self.time_step = 0
        if not self.with_goal:
            self.set_hidden_goal()

        goal_state = self._sample_goal_state()
        if goal_state is None:
            self.goal_state = None
        else:
            self.goal_state = goal_state.copy()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.last_observation = self._get_obs()
        return self.last_observation

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.use_auxiliary_info:
            # Save current image
            prev_frame = self.last_observation.copy()['observation']

            # Perform env step
            self._set_action(action)
            self.sim.step()
            self._step_callback()
            obs = self._get_obs()

            # Prepare auxiliary information
            transformed_img, transformation = self.random_image_transformation(prev_frame)
            prev_frame = prev_frame[:, :, -4:]  # For stacked observation
            op_flow_image = self.get_optical_flow_image(prev_frame, obs['observation'][:, :, -4:])
            bw_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
            aug_info = {
                'bw_frame': bw_frame.flatten(),
                'transformed_frame': transformed_img.flatten(),
                'transformation': transformation,
                'op_flow': op_flow_image.flatten(),
            }
        else:
            self._set_action(action)
            self.sim.step()
            self._step_callback()
            obs = self._get_obs()
            aug_info = {}

        self.last_observation = obs.copy()
        state_info = self.get_current_info()
        info = {**aug_info, **state_info}

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        self.time_step += 1
        # Episode ends only when the horizon is reached
        done = False
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info

    def get_initial_info(self):
        state_info = self.get_current_info()
        obs_shape = self.observation_space.spaces['observation'].shape
        if self.use_auxiliary_info:
            aug_info = {
                'transformed_frame': np.zeros(obs_shape, dtype=np.float).flatten(),
                'transformation': np.array([0., 0., 0.]),
                'bw_frame': np.zeros([self.image_size, self.image_size], dtype=np.float).flatten(),
                'op_flow': np.zeros([self.image_size, self.image_size], dtype=np.float).flatten(),
            }
            return {**aug_info, **state_info}
        else:
            return state_info

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    def render(self, mode='human', image_size=None, depth=True, camera_name=None):
        self._render_callback()
        if camera_name is None:
            camera_name = self.default_camera_name
        if image_size is None:
            image_size = self.image_size
        if mode == 'rgb_array':
            self.sim.render_contexts[0]._set_mujoco_buffers()
            if depth:
                image, depth = self.sim.render(camera_name=camera_name, width=image_size, height=image_size, depth=True)
                # id = self.sim.model.camera_name2id('external_camera_0')
                # print(self.sim.model.cam_fovy)
                rgbd_data = np.dstack([image, depth])
                return rgbd_data[::-1, :, :]
            else:
                image = self.sim.render(camera_name=camera_name, width=image_size, height=image_size, depth=False)
                return image[::-1, :, :]
        elif mode == 'human':
            # return self._get_viewer().render()
            image = self.sim.render(camera_name='external_camera_0', width=image_size, height=image_size,
                                    depth=False)
            cv_render(image)

    # Auxiliary tasks info
    # ----------------------------

    @staticmethod
    def random_image_transformation(image, max_translation=10, max_angle=30):
        '''
        Randomly translate and rotation the image
        :param image: image to be transformed
        :param max_translation: max translation in pixels
        :param max_angle: max rotation angles in degrees
        :return: The transformed image and the transformation params
        '''
        angle = np.random.uniform(-max_angle, max_angle)
        translation_x = np.random.uniform(-max_translation, max_translation)
        translation_y = np.random.uniform(-max_translation, max_translation)

        height, width = image.shape[0], image.shape[1]
        M1 = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        M2 = np.float32([[0, 0, translation_x], [0, 0, translation_y]])

        transformed_img = cv.warpAffine(image, M1 + M2, (image.shape[1], image.shape[0]))
        return transformed_img, np.asarray([float(angle) / max_angle, float(translation_x) / max_translation, float(translation_y) / max_translation])

    @staticmethod
    def get_optical_flow_image(prev_frame, next_frame):
        ''' Return an image indicating the magnitude of the flow'''
        if prev_frame.shape[2] == 4:  # RGBD image
            prev_frame = prev_frame[:, :, :3]
            next_frame = next_frame[:, :, :3]
        prev_frame = prev_frame.copy() / 255.
        next_frame = next_frame.copy() / 255.
        prvs = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(prev_frame)
        next_im = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next_im, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2  # Direction of flow
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # Magnitude of flow
        return hsv[..., 2]

    # Helper Functions
    # ----------------------------
    def _get_info_state(self, achieved_goal, desired_goal):
        # Given g, ag in state space and return the distance and success
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return d, (d <= self.distance_threshold).astype(np.float32)

    def _get_info_obs(self, achieved_goal_obs, desired_goal_obs):
        # Given g, ag in state space and return the distance and success
        achieved_goal_obs = achieved_goal_obs.reshape([-1, self.goal_dim])
        desired_goal_obs = desired_goal_obs.reshape([-1, self.goal_dim])
        d = np.linalg.norm(achieved_goal_obs - desired_goal_obs, axis=-1)
        return d, (d <= self.distance_threshold_obs).astype(np.float32)

    def set_camera_location(self, camera_name=None, pos=[0.0, 0.0, 0.0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_pos[id] = pos

    def set_camera_fov(self, camera_name=None, fovy=50.0):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_fovy[id] = fovy

    def set_camera_orientation(self, camera_name=None, orientation_quat=[0, 0, 0, 0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_quat[id] = orientation_quat
