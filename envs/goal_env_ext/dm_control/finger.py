from gym import utils
import numpy as np
from dm_control.rl.control import PhysicsError
from dm_control.suite.utils import randomizers
from termcolor import colored
from dm_control.utils import containers
from envs.goal_env_ext.dm_control.dm_control_env import DmControlEnv
from envs.goal_env_ext.dm_control import common

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02  # (seconds)
# For TURN tasks, the 'tip' geom needs to enter a spherical target of sizes:
_EASY_TARGET_SIZE = 0.00
_HARD_TARGET_SIZE = 0.03
# Initial spin velocity for the Stop task.
_INITIAL_SPIN_VELOCITY = 100
# Spinning slower than this value (radian/second) is considered stopped.
_STOP_VELOCITY = 1e-6
# Spinning faster than this value (radian/second) is considered spinning.
_SPIN_VELOCITY = 15.0

SUITE = containers.TaggedTasks()

from collections import deque


class Finger(DmControlEnv, utils.EzPickle):
    def __init__(self, model_path='finger.xml', n_substeps=2, n_actions=2, goal_range=[-2.0, 2.0], stack_obs=False, **kwargs):
        # horizon = 1000, goal_range = [-2.0, 2.0], image_size = 460, init_position = 'goal_range', use_auxiliary_loss = False, use_visual_observation = True,
        # noisy_reward_fp = False, noisy_reward_fn = False, use_true_reward = True,distance_threshold_obs = 0.0, **kwargs):

        """
        :param model_path:
        :param distance_threshold:
        :param frame_skip:
        :param horizon:
        :param goal_range:
        :param action_type: Should be in ['force', 'velocity', 'position']
        :param image_size:
        """
        self.n_substeps = n_substeps
        self._target_radius = _EASY_TARGET_SIZE
        self.stack_obs = stack_obs
        self.stack_buffer = deque(maxlen=3)
        DmControlEnv.__init__(
            self, model_path, n_substeps=n_substeps, n_actions=n_actions, initial_qpos=None, default_camera_name='cam0',
            **kwargs)

        utils.EzPickle.__init__(self)

        self.goal_range = goal_range

    def get_model_and_assets(self):
        """Returns a tuple containing the model XML string and a dict of assets.

        Args:
          n_joints: An integer specifying the number of joints in the swimmer.

        Returns:
          A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
          `{filename: contents_string}` pairs.
        """

        return common.read_model('finger.xml'), common.ASSETS

    def touch(self):
        """Returns logarithmically scaled signals from the two touch sensors."""
        return np.log1p(self.physics.named.data.sensordata[['touchtop', 'touchbottom']])

    def hinge_velocity(self):
        """Returns the velocity of the hinge joint."""
        return self.physics.named.data.sensordata['hinge_velocity']

    def tip_position(self):
        """Returns the (x,z) position of the tip relative to the hinge."""
        return (self.physics.named.data.sensordata['tip'][[0, 2]] -
                self.physics.named.data.sensordata['spinner'][[0, 2]])

    def bounded_position(self):
        """Returns the positions, with the hinge angle replaced by tip position."""
        return np.hstack((self.physics.named.data.sensordata[['proximal', 'distal']],
                          self.tip_position()))

    def velocity(self):
        """Returns the velocities (extracted from sensordata)."""
        return self.physics.named.data.sensordata[['proximal_velocity', 'distal_velocity', 'hinge_velocity']]

    def target_position(self):
        """Returns the (x,z) position of the target relative to the hinge."""
        return (self.physics.named.data.sensordata['target'][[0, 2]] -
                self.physics.named.data.sensordata['spinner'][[0, 2]])

    def to_target(self):
        """Returns the vector from the tip to the target."""
        return self.target_position() - self.tip_position()

    def dist_to_target(self):
        """Returns the signed distance to the target surface, negative is inside."""
        return (np.linalg.norm(self.to_target()) -
                self.physics.named.model.site_size['target', 0])

    def _reset_sim(self):
        self.time_step = 0
        with self.physics.reset_context():
            target_angle = self.random.uniform(-np.pi, np.pi)
            self._set_to_goal(target_angle)

            hinge_x, hinge_z = self.physics.named.data.xanchor['hinge', ['x', 'z']]
            radius = self.physics.named.model.geom_size['cap1'].sum()
            target_x = hinge_x + radius * np.sin(-target_angle)
            target_z = hinge_z + radius * np.cos(-target_angle)
            self.physics.named.model.site_pos['target', ['x', 'z']] = target_x, target_z
            self.physics.named.model.site_size['target', 0] = self._target_radius
            # self.physics.named.model.site_size['tip', 0] = 0.02
            self.physics.named.model.site_size['tip', 0] = 0.05
        self.goal_image = self.render()
        with self.physics.reset_context():
            self._set_random_joint_angles(self.physics, self.random)
        self.goal_state = self.target_position().copy()
        self.stack_buffer.clear()
        return True

    def _set_to_goal(self, target_angle):
        with self.physics.reset_context():
            self.physics.named.data.qpos['hinge'] = target_angle

    def _set_random_joint_angles(self, physics, random, max_attempts=1000):
        """Sets the joints to a random collision-free state."""
        for _ in range(max_attempts):
            randomizers.randomize_limited_and_rotational_joints(physics, random)
            # Check for collisions.
            physics.after_reset()
            if physics.data.ncon == 0:
                break
        else:
            raise RuntimeError('Could not find a collision-free state '
                               'after {} attempts'.format(max_attempts))

    # def _get_obs_stack(self, image_size=None, camera_name=None, depth=True):
    #     try:
    #         for _ in range(int(self.n_substeps / 3) + 1):
    #             self.physics.step()
    #     except PhysicsError as ex:
    #         print(colored(ex, 'red'))
    #
    #     ob1 = self.render().copy()
    #
    #     try:
    #         for _ in range(int(self.n_substeps / 3) + 1):
    #             self.physics.step()
    #     except PhysicsError as ex:
    #         print(colored(ex, 'red'))
    #
    #     ob2 = self.render().copy()
    #
    #     try:
    #         for _ in range(int(self.n_substeps / 3) + 1):
    #             self.physics.step()
    #     except PhysicsError as ex:
    #         print(colored(ex, 'red'))
    #
    #     ob3 = self.render().copy()
    #     obs = np.dstack((ob1, ob2, ob3))
    #     return {
    #         'observation': obs.copy(),
    #         'achieved_goal': self.tip_position().copy(),
    #         'desired_goal': self.target_position().copy()
    #     }

    def _get_obs(self):
        # traceback.print_stack()
        if self.use_visual_observation:
            if self.stack_obs:
                steps = [x[1] for x in self.stack_buffer]
                if len(self.stack_buffer) > 0 and self.time_step == self.stack_buffer[-1][1]:
                    obs = np.concatenate([ob[0] for ob in self.stack_buffer], axis=-1)
                else:
                    try:
                        for _ in range(int(self.n_substeps) + 1):
                            self.physics.step()
                    except PhysicsError as ex:
                        print(colored(ex, 'red'))
                    new_obs = self.render()
                    self.stack_buffer.append((new_obs.copy(), self.time_step))
                    # Fill the buffer
                    while len(self.stack_buffer) < self.stack_buffer.maxlen:
                        self.stack_buffer.append((new_obs.copy(), self.time_step))
                    obs = np.concatenate([ob[0] for ob in self.stack_buffer], axis=-1)
            else:
                try:
                    for _ in range(int(self.n_substeps) + 1):
                        self.physics.step()
                except PhysicsError as ex:
                    print(colored(ex, 'red'))

                obs = self.render().copy()
        else:
            assert False
            obs = np.concatenate((self.bounded_position().flatten().copy(), self.velocity().flatten().copy(),
                                  self.touch().flatten().copy(),
                                  self.target_position().flatten().copy(), self.dist_to_target().flatten().copy()))

        if self.use_visual_observation and self.stack_buffer:
            achieved_goal = np.tile(obs[:, :, -4:], [1, 1, 3])
            desired_goal = np.tile(self.goal_image, [1, 1, 3])
        else:
            achieved_goal = obs.copy()
            desired_goal = self.goal_image.copy()
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }

    def get_current_info(self):

        info = {
            'is_success': self._is_success(self.tip_position().copy(), self.target_position().copy()),
            'ag_state': self.tip_position().copy(),
            'g_state': self.target_position().copy()
        }

        return info

    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1) - self.physics.named.model.site_size['target', 0]
        return (d <= self.distance_threshold).astype(np.float32)

    def _get_info_state(self, achieved_goal, desired_goal):
        # Given g, ag in state space and return the distance and success
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1) - self.physics.named.model.site_size['target', 0]
        return d, (d < self.distance_threshold).astype(np.float32)
