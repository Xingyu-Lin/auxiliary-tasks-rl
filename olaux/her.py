import numpy as np


def make_sample_her_transitions(replay_strategy,
                                replay_k,
                                reward_fun,
                                reward_choices=(-1, 1),
                                use_aux_tasks=False):
    """ Creates a sample function that can be used for HER experience replay.
    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
        reward_choices: Rewards when goal is achieved / not achieved
        use_aux_tasks: Whether auxiliary information will also be sampled in the transitions
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        episode_batch['info_ag_2_state'] = episode_batch['info_ag_state'][:, 1:, :]
        T = episode_batch['u'].shape[1]
        if use_aux_tasks:
            episode_batch['info_transformation'] = episode_batch['info_transformation'][:, 1:, :]
            episode_batch['info_transformed_frame'] = episode_batch['info_transformed_frame'][:, 1:, :]
            episode_batch['info_op_flow'] = episode_batch['info_op_flow'][:, 1:, :]
            episode_batch['info_bw_frame'] = episode_batch['info_bw_frame'][:, 1:, :]

        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        # Baseline HER with a replay probability
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)

        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        future_ag_val = episode_batch['info_ag_state'][episode_idxs[her_indexes], future_t]
        transitions['info_g_state'][her_indexes] = future_ag_val
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        info['ag_state'] = info['ag_2_state']

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        rewards = reward_fun(**reward_params)
        goal_reached = (np.round(rewards) == 0)  # Make sure the environments are zero when goals are reached
        rewards = goal_reached * reward_choices[1] + (1. - goal_reached) * reward_choices[0]
        transitions['r'] = rewards

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
