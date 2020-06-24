from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from olaux.utils import convert_episode_to_batch_major, store_args
import cv2 as cv


def shapes_to_dims(input_shapes):
    return {key: np.prod(val) for key, val in input_shapes.items()}


class RolloutWorker:
    @store_args
    def __init__(self, make_env, policy, shapes, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, full_info=True, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        dims = shapes_to_dims(shapes)
        self.dims = dims
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.logging_keys = ['success_state', 'success_obs', 'goal_dist_final_state', 'goal_dist_final_obs']
        if self.compute_Q:
            self.logging_keys.append('mean_Q')
        # Smoothed by maxlen
        self.logging_history = {key: deque(maxlen=history_len) for key in self.logging_keys}
        # Logging history: [T, batch_id]
        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.full_info = full_info
        if full_info:
            self.initial_info = [None] * rollout_batch_size
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        if self.full_info:
            if hasattr(self.envs[i], 'get_initial_info'):
                self.initial_info[i] = self.envs[i].get_initial_info()
            else:
                # Only used for visualization of deprecated policy
                self.initial_info[i] = self.envs[i].get_current_info()

        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation'].flatten()
        self.initial_ag[i] = obs['achieved_goal'].flatten()
        self.g[i] = obs['desired_goal'].flatten()

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals = [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in
                       self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)
            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output
            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])

                    o_new[i] = curr_o_new['observation'].flatten()
                    ag_new[i] = curr_o_new['achieved_goal'].flatten()
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render == 1 and i==0:
                        self.envs[i].render()
                    elif self.render == 2 and i==0:
                        display_o = cv.resize(curr_o_new['observation'], (500, 500))
                        display_g = cv.resize(curr_o_new['desired_goal'], (500, 500))
                        display_img = np.concatenate((display_o, display_g), axis=1)
                        display_img = display_img[:, :, (2, 1, 0)] / 256.
                        cv.imshow('display', display_img)
                        cv.waitKey(10)
                except MujocoException as e:
                    return self.generate_rollouts()
            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        if self.full_info:
            for idx, key in enumerate(self.info_keys):
                init_info_values = [np.empty((1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key
                                    in self.info_keys]
                for t_idx, t_key in enumerate(self.info_keys):
                    for i in range(self.rollout_batch_size):
                        init_info_values[t_idx][0, i] = self.initial_info[i][t_key]
                info_values[idx] = np.concatenate([init_info_values[idx], info_values[idx]], axis=0)

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        d, d_bool = self.envs[0]._get_info_state(episode['info_ag_state'][-1][:], episode['info_g_state'][-1][:])
        self.logging_history['goal_dist_final_state'].append(np.mean(d))
        self.logging_history['success_state'].append(np.mean(d_bool))

        d, d_bool = self.envs[0]._get_info_obs(episode['ag'][-1][:], episode['g'][-1][:])
        self.logging_history['goal_dist_final_obs'].append(np.mean(d))
        self.logging_history['success_obs'].append(np.mean(d_bool))

        if self.compute_Q:
            self.logging_history['mean_Q'].append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        for _, log_queue in self.logging_history.items():
            log_queue.clear()

    def current_success_rate(self):
        return np.mean(self.logging_history['success_state'])

    def current_mean_Q(self):
        return np.mean(self.logging_history['mean_Q'])

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        # print(dir(self.policy))
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []

        for key, log_queue in sorted(self.logging_history.items()):
            logs += [(key, np.mean(self.logging_history[key]))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
