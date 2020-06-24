import json
import logging
import argparse
from mpi4py import MPI

from olaux.rollout import RolloutWorker

from olaux import logger, config
from olaux.utils import *

# Remove annoying tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='OL-AUX', type=str)  # Experiment Name
    parser.add_argument('--env_name', default='VisualFetchReach')  # Task name: ['VisualFetchReach', 'VisualHandReach', 'VisualFinger']

    # train
    parser.add_argument('--num_cpu', default=8, type=int)  # Number of CPU workers. Caution in changing this as many other parameters are related.
    parser.add_argument('--scale_grad_by_procs', action='store_true')  # Whether the gradients are scaled by the number of CPUs
    parser.add_argument('--batch_size', default=64, type=int)  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    parser.add_argument('--rollout_batch_size', default=10, type=int)  # Number of trajectories collected per cycle per CPU worker
    parser.add_argument('--n_batches', default=40, type=int)  # Number of training loop per cycle

    # eval
    parser.add_argument('--n_test_rollouts', default=2)  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts per thread
    parser.add_argument('--test_with_polyak', action='store_true')

    # DDPG
    parser.add_argument('--buffer_size', default=5000,
                        type=int)  # Replay buffer size. Each CPU worker has a separate buffer so default size is 5000 x 8
    parser.add_argument('--Q_lr', default=1e-3, type=float)  # critic learning rate
    parser.add_argument('--pi_lr', default=1e-3, type=float)  # actor learning rate
    parser.add_argument('--polyak', default=0.98, type=float)  # polyyak average coefficient
    parser.add_argument('--layers', default=3, type=int)  # number of layers in the critic/actor networks
    parser.add_argument('--hidden', default=256, type=int)  # number of neurons in each hidden layers
    parser.add_argument('--max_u', default=1.0, type=float)  # max absolute value of actions on different coordinates
    parser.add_argument('--action_l2', default=1.0, type=float)  # quadratic penalty on actions (before rescaling by max_u)
    parser.add_argument('--scope', default='ddpg', type=str)  # Scope for TF graph
    parser.add_argument('--norm_eps', default=0.01, type=float)  # epsilon used for observation normalization
    parser.add_argument('--norm_clip', default=5., type=float)  # normalized observations are cropped to this values
    parser.add_argument('--random_eps', default=0.3, type=float)  # percentage of time a random action is taken for exploration
    parser.add_argument('--noise_eps', default=0.2,
                        type=float)  # std of gaussian noise added to not-completely-random actions as a percentage of max_u

    # encoder
    parser.add_argument('--cnn_nonlinear', default='relu', type=str)
    parser.add_argument('--use_bottleneck_layer', default=1, type=int)
    parser.add_argument('--dim_latent_repr', default=512, type=int)

    # Hindsight experience replay params
    parser.add_argument('--her_replay_strategy', default='future', type=str)
    parser.add_argument('--her_reward_choices', default=(-1, 0), type=tuple)
    parser.add_argument('--her_replay_k', default=9, type=int)

    # OL-AUX
    parser.add_argument('--use_aux_tasks', default=1, type=int)  # Whether auxiliary tasks are used
    parser.add_argument('--aux_update_interval', default=5, type=int)  # How often we update auxiliary params
    parser.add_argument('--aux_base_lr', default=5, type=float)  # Leanring rate for auxiliary weights
    parser.add_argument('--aux_filter_interval', default=4, type=int)  # In case the learning rate is too large, the gradients for both aux tasks
    # and main tasks may occilicate. This is to smooth the graidents before taking their dot product
    parser.add_argument('--log_loss', default=1, type=int)  # Whether gradient balancing is used
    parser.add_argument('--log_min_loss', default=0.02, type=float)  # Minimum loss before applying the log (For stability)

    # misc
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--policy_save_interval', default=10, type=int)
    parser.add_argument('--save_policies', default=1, type=int)
    parser.add_argument('--visualize_training', action='store_true')

    args = parser.parse_args()
    return args


def run_task(vv):
    # Fork for multi-CPU MPI implementation.
    if vv['num_cpu'] > 1:
        whoami = mpi_fork(vv['num_cpu'])
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        logdir = os.path.join('./data', vv['exp_name'])
        logger.configure(logdir, exp_name=vv['exp_name'])
        os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = vv['seed'] + 1000000 * rank
    set_global_seeds(rank_seed)

    # Environment arguments for custom environments
    env_name = vv['env_name']
    env_kwargs = vv['env_kwargs'] = config.env_arg_dict[env_name]

    # Update variants
    vv['n_epochs'] = env_kwargs['n_epochs']  # Number of epochs to train
    vv['n_cycles'] = env_kwargs['n_cycles']  # Each cycle include an exploration phase and a training from buffer phase.
    vv['T'] = env_kwargs['horizon']
    vv['max_u'] = np.array(vv['max_u']) if type(vv['max_u']) == list else vv['max_u']
    vv['gamma'] = 1. - 1. / vv['T']
    vv['her_use_aux_tasks'] = vv['env_kwargs']['use_auxiliary_info'] = vv['use_aux_tasks']

    if rank == 0:
        # Dump parameters
        with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
            json.dump(vv, f, indent=2, sort_keys=True)
        log_params(vv, logger)

    def make_env():
        return config.CUSTOM_ENVS[env_name](**env_kwargs)

    vv['make_env'] = make_env

    # Create policy and rollout workers
    shapes = config.configure_shapes(vv['make_env'])
    dims = shapes_to_dims(shapes)
    policy = config.configure_ddpg(dims=dims, shapes=shapes, vv=vv, rank=rank)

    rollout_worker = RolloutWorker(vv['make_env'], policy, dims, logger,
                                   T=vv['T'],
                                   rollout_batch_size=vv['rollout_batch_size'],
                                   gamma=vv['gamma'],
                                   noise_eps=vv['noise_eps'],
                                   random_eps=vv['random_eps'],
                                   exploit=False,
                                   use_target_net=False,
                                   compute_Q=False)

    # Visualize exploration
    if rank == 0 and vv['visualize_training']:
        rollout_worker.render = 2

    evaluator = RolloutWorker(vv['make_env'], policy, dims, logger,
                              T=vv['T'],
                              rollout_batch_size=vv['rollout_batch_size'],
                              gamma=vv['gamma'],
                              noise_eps=vv['noise_eps'],
                              random_eps=vv['random_eps'],
                              exploit=True,
                              use_target_net=vv['test_with_polyak'],
                              compute_Q=True)
    rollout_worker.seed(rank_seed)
    evaluator.seed(rank_seed)

    if rank == 0:
        latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
        best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
        periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

        logger.info('Saving tf graph...')
        import tensorflow as tf
        tf.summary.FileWriter(logger.get_dir(), policy.sess.graph)

    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(vv['n_epochs']):
        # train
        rollout_worker.clear_history()
        for i_cycle in range(vv['n_cycles']):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)

            for i_batch in range(vv['n_batches']):
                critic_loss, actor_loss = policy.train()
            policy.update_target_net()
        # eval
        evaluator.clear_history()
        for i_test_cycle in range(vv['n_test_rollouts']):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('loss/actor_loss', actor_loss)
        logger.record_tabular('loss/critic_loss', critic_loss)

        for key, val in evaluator.logs('test'):
            if 'episode' in key:  # Include the episodes from all workers
                logger.record_tabular(key, vv['num_cpu'] * mpi_average(val))
            else:
                logger.record_tabular(key, mpi_average(val))

        for key, val in rollout_worker.logs('train'):
            if 'episode' in key:
                logger.record_tabular(key, vv['num_cpu'] * mpi_average(val))
            else:
                logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        success_rate = mpi_average(evaluator.current_success_rate())

        if rank == 0:
            logger.dump_tabular()
            # save the policy if it's better than the previous ones
            if success_rate >= best_success_rate and vv['save_policies']:
                best_success_rate = success_rate
                logger.info(
                    'New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
                evaluator.save_policy(best_policy_path)
                evaluator.save_policy(latest_policy_path)
            if epoch % vv['policy_save_interval'] == 0 and vv['save_policies']:
                policy_path = periodic_policy_path.format(epoch)
                logger.info('Saving periodic policy to {} ...'.format(policy_path))
                evaluator.save_policy(policy_path)
        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


if __name__ == '__main__':
    vv = parse_args().__dict__
    run_task(vv)
