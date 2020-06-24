import numpy as np

from olaux.ddpg import DDPG
from olaux.her import make_sample_her_transitions

from envs.goal_env_ext.fetch.reach import FetchReachEnv
from envs.goal_env_ext.hand.reach import HandReachEnv
from envs.goal_env_ext.dm_control.finger import Finger

CUSTOM_ENVS = {
    'VisualFetchReach': FetchReachEnv,
    'VisualHandReach': HandReachEnv,
    'VisualFinger': Finger
}

env_arg_dict = {
    'VisualFetchReach': {'n_cycles': 5, 'n_epochs': 40, 'horizon': 50, 'image_size': 100, 'distance_threshold': 5e-2,
                         'distance_threshold_obs': 0.,
                         'use_image_goal': True, 'use_visual_observation': True, 'with_goal': False},
    'VisualHandReach': {'n_cycles': 10, 'n_epochs': 150, 'horizon': 50, 'image_size': 100, 'distance_threshold': 3e-2,
                        'distance_threshold_obs': 0., 'use_image_goal': True, 'use_visual_observation': True,
                        'with_goal': False},
    'VisualFinger': {'n_cycles': 10, 'n_epochs': 100, 'horizon': 50, 'distance_threshold': 0.07, 'distance_threshold_obs': 0., 'n_substeps': 10,
                     'image_size': 100, 'use_auxiliary_info': True, 'use_image_goal': True, 'use_visual_observation': True,
                     'stack_obs': False}
}

CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def configure_her(vv):
    env = cached_make_env(vv['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    her_vv = {'reward_fun': reward_fun}
    for name in vv:
        if name.startswith('her_'):
            her_vv[name[len('her_'):]] = vv[name]
    sample_her_transitions = make_sample_her_transitions(**her_vv)

    return sample_her_transitions


def configure_ddpg(vv, dims, shapes, reuse=False, clip_return=True, rank=0):
    sample_her_transitions = configure_her(vv)
    ddpg_vv = vv.copy()
    ddpg_vv.update({'input_dims': dims.copy(),  # agent takes an input observations
                    'image_input_shapes': shapes.copy(),
                    'clip_pos_returns': False,  # clip positive returns
                    'clip_return': (1. / (1. - vv['gamma'])) if clip_return else np.inf,  # max abs of return
                    'sample_transitions': sample_her_transitions,
                    'rank': rank})

    policy = DDPG(reuse=reuse, **ddpg_vv)
    return policy


def configure_shapes(make_env):
    env = cached_make_env(make_env)
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    shapes = {
        'o': obs['observation'].shape,
        'u': env.action_space.shape,
        'g': obs['desired_goal'].shape,
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        shapes['info_{}'.format(key)] = value.shape
    return shapes
