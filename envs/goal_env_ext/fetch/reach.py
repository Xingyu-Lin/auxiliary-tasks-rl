from gym import utils
from envs.goal_env_ext.fetch import fetch_env


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', distance_threshold=0.05, n_substeps=20, **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/reach.xml', has_object=False, block_gripper=True, n_substeps=n_substeps,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type, **kwargs)
        utils.EzPickle.__init__(self)
