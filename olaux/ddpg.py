from collections import OrderedDict
import copy
from tensorflow.contrib.staging import StagingArea

from olaux import logger
from olaux.normalizer import Normalizer
from olaux.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_aux_update import MpiAuxUpdate
from collections import deque
from olaux.utils import *


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnresolvedReferences
class DDPG(object):
    @store_args
    def __init__(self,
                 use_aux_tasks,
                 input_dims,
                 image_input_shapes,
                 buffer_size,
                 hidden,
                 layers,
                 dim_latent_repr,
                 cnn_nonlinear,
                 use_bottleneck_layer,
                 polyak,
                 batch_size,
                 Q_lr,
                 pi_lr,
                 norm_eps,
                 norm_clip,
                 max_u,
                 action_l2,
                 scope,
                 T,
                 rollout_batch_size,
                 clip_pos_returns,
                 clip_return,
                 log_loss,
                 sample_transitions,
                 gamma,
                 rank,
                 serialized=False,
                 reuse=False,
                 clip_grad_range=None,
                 aux_filter_interval=None,
                 scale_grad_by_procs=False,
                 aux_update_interval=5,
                 aux_base_lr=5,
                 **kwargs):
        """ See the documentation in main.py """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function('cnn_actor_critic:CNNActorCritic')

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        if self.use_aux_tasks:
            self.dim_bw_frame = self.input_dims['info_bw_frame']
            self.dim_op_flow = self.input_dims['info_op_flow']
            self.dim_transformed_frame = self.input_dims['info_transformed_frame']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()

        include_info = ['info_state_obs', 'info_transformed_frame', 'info_transformation',
                        'info_op_flow', 'info_bw_frame']

        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_') and not key in include_info:
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            if self.use_aux_tasks:
                # Initialize OL-AUX
                self.num_auxiliary_tasks = 5
                self.aux_weights_lr = self.aux_base_lr * self.aux_update_interval

                self.aux_weight_vector_Q_tf = tf.Variable(initial_value=1 * tf.ones(self.num_auxiliary_tasks),
                                                          dtype=tf.float32, name='aux_weights')
                self.aux_weight_grads_buffer = []
                self.log_aux_losses_Q = self.log_aux_tasks_losses_pi = None  # Logging buffer for aux losses
                if self.aux_filter_interval is not None:
                    self.all_grad_history = deque(maxlen=self.aux_filter_interval)

            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=self.reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' and not key.startswith('info_') else self.T + 1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T + 1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }
        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (
          self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = o.copy(), g.copy()
            # No need to preprocess the o_2 and g_2 since this is only used for stats
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

            if self.use_aux_tasks:
                self.bw_frame_stats.update(transitions['info_bw_frame'])
                self.op_flow_stats.update(transitions['info_op_flow'])
                self.transformed_frame_stats.update(transitions['info_transformed_frame'])
                self.bw_frame_stats.recompute_stats()
                self.op_flow_stats.recompute_stats()
                self.transformed_frame_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        assert not self.serialized
        run_vars = [self.Q_loss_tf, self.pi_loss_tf, self.Q_grad_tf, self.pi_grad_tf]

        if self.use_aux_tasks:
            run_vars.append(self.main_task_Q_cnn_grad_flatten_tf)
            run_vars.extend(self.main.loss_auxiliary_tasks_Q_tf)  # Q Aux losses
            run_vars.extend(self.aux_Q_cnn_grads_flatten_tf)  # Q Aux grads
            run_vars.extend(self.main.loss_auxiliary_tasks_pi_tf)  # pi Aux losses
            assert len(self.aux_Q_cnn_grads_flatten_tf) == self.num_auxiliary_tasks
            rets = self.sess.run(run_vars)

            aux_losses_pi = copy.copy(rets[-self.num_auxiliary_tasks:])
            aux_grads_Q = copy.copy(rets[-2 * self.num_auxiliary_tasks:-self.num_auxiliary_tasks])
            aux_losses_Q = copy.copy(rets[-3 * self.num_auxiliary_tasks:-2 * self.num_auxiliary_tasks])

            rets = rets[:-3 * self.num_auxiliary_tasks] + [aux_losses_pi] + [aux_losses_Q] + [aux_grads_Q]
        else:
            rets = self.sess.run(run_vars)
        return rets

    # noinspection PyAttributeOutsideInit
    def train(self, stage=True):
        # import cProfile, pstats, io
        # pr = cProfile.Profile()
        # pr.enable()
        if stage:
            self.stage_batch()
        if self.use_aux_tasks:
            critic_loss, actor_loss, Q_grad, pi_grad, main_task_grad, \
            aux_losses_pi, aux_losses_Q, aux_task_grads_Q = self._grads()

            self.log_aux_losses_Q = [loss for loss in aux_losses_Q]
            self.log_aux_losses_pi = [loss for loss in aux_losses_pi]

            self._update(Q_grad, pi_grad)
            self._update_aux_weights(main_task_grad, aux_task_grads_Q)
        else:
            critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
            self._update(Q_grad, pi_grad)
        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('time')
        # ps.print_stats(20)
        # print(s.getvalue())

        return critic_loss, actor_loss

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        transitions['o'], transitions['g'] = o.copy(), g.copy()
        transitions['o_2'], transitions['g_2'] = o_2.copy(), g.copy()

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))


    def _update_aux_weights(self, main_task_grad, aux_task_grads):
        """
        Called during each iteration. But only update the auxiliary task weights according to the update interval
        :param main_task_grad: Gradient of the main task (of cnn)
        :param aux_task_grads: A list of the gradients from each of the auxiliary tasks (of cnn)
        """
        main_task_grad, aux_task_grads = self.aux_weight_updater.get_syncd_grad(main_task_grad, aux_task_grads)

        aux_weight_grad = np.zeros([self.num_auxiliary_tasks])
        aux_task_grads = np.array(aux_task_grads)
        main_task_grad = np.array(main_task_grad)

        if self.aux_filter_interval is not None:
            self.all_grad_history.append((main_task_grad.copy(), aux_task_grads.copy()))
            main_task_grad = np.mean(np.array([grad[0] for grad in self.all_grad_history]), axis=0)
            aux_task_grads = np.mean(np.array([grad[1] for grad in self.all_grad_history]), axis=0)

        for i, aux_task_grad in enumerate(aux_task_grads):
            aux_weight_grad[i] = self.Q_lr * np.dot(aux_task_grad, main_task_grad)
        self.aux_weight_grads_buffer.append(aux_weight_grad)

        if len(self.aux_weight_grads_buffer) == self.aux_update_interval:
            aggregate_aux_weight_grad = np.mean(np.array(self.aux_weight_grads_buffer), axis=0)
            self.aux_weight_updater.update(self.aux_weights_lr * aggregate_aux_weight_grad)
            self.aux_weight_grads_buffer = []

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)

        if self.use_aux_tasks:
            with tf.variable_scope('bw_frame_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.bw_frame_stats = Normalizer(self.dim_bw_frame, self.norm_eps, self.norm_clip, sess=self.sess)

            with tf.variable_scope('op_flow_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.op_flow_stats = Normalizer(self.dim_op_flow, self.norm_eps, self.norm_clip, sess=self.sess)

            with tf.variable_scope('transformed_frame_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.transformed_frame_stats = Normalizer(self.dim_transformed_frame, self.norm_eps, self.norm_clip,
                                                          sess=self.sess)

        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            if self.use_aux_tasks:
                self.main.build_auxiliary_tasks()
            vs.reuse_variables()

        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)

        if self.use_aux_tasks and self.log_loss:
            self.pi_loss_tf = tf.clip_by_value(self.pi_loss_tf, np.finfo(float).eps, np.Inf)  # So that log can be applied
            self.Q_loss_tf = tf.log(self.Q_loss_tf)
            self.pi_loss_tf = tf.log(self.pi_loss_tf)

        self.action_l2_loss_tf = self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.pi_loss_tf += self.action_l2_loss_tf

        if self.use_aux_tasks:
            if self.log_loss:
                for i, loss_tf in enumerate(self.main.loss_auxiliary_tasks_Q_tf):
                    self.Q_loss_tf += tf.stop_gradient(self.aux_weight_vector_Q_tf[i]) * tf.log(loss_tf + self.log_min_loss)

                # Use the same weight of the auxiliary tasks in Q function also for pi.
                # Also possible to use separate aux weight vectors for Q and pi
                for i, loss_tf in enumerate(self.main.loss_auxiliary_tasks_pi_tf):
                    self.pi_loss_tf += tf.stop_gradient(self.aux_weight_vector_Q_tf[i]) * tf.log(loss_tf + self.log_min_loss)
            else:
                for i, loss_tf in enumerate(self.main.loss_auxiliary_tasks_Q_tf):
                    self.Q_loss_tf += tf.stop_gradient(self.aux_weight_vector_Q_tf[i]) * loss_tf
                for i, loss_tf in enumerate(self.main.loss_auxiliary_tasks_pi_tf):
                    self.pi_loss_tf += tf.stop_gradient(self.aux_weight_vector_Q_tf[i]) * loss_tf

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'), name='Q_gradient')
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'), name='pi_gradient')

        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'), clip_grad_range=self.clip_grad_range)
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'), clip_grad_range=self.clip_grad_range)

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=self.scale_grad_by_procs)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=self.scale_grad_by_procs)
        if self.use_aux_tasks:
            self.aux_weight_updater = MpiAuxUpdate(self._vars('aux_weights'), scale_grad_by_procs=True)

        if self.use_aux_tasks:
            # Get gradient from the auxiliary tasks w.r.t. the shared cnn
            if self.log_loss:
                aux_Q_cnn_grads_tf = [tf.gradients(tf.log(loss_tf + self.log_min_loss, name=loss_name), self._vars('main/Q/cnn')) for
                                  (loss_tf, loss_name) in zip(self.main.loss_auxiliary_tasks_Q_tf, self.main.name_auxiliary_tasks)]
            else:
                aux_Q_cnn_grads_tf = [tf.gradients(loss_tf, self._vars('main/Q/cnn')) for loss_tf in self.main.loss_auxiliary_tasks_Q_tf]
            self.aux_Q_cnn_grads_flatten_tf = [
                flatten_grads(grads=aux_grad_tf, var_list=self._vars('main/Q/cnn'), clip_grad_range=self.clip_grad_range) for
                aux_grad_tf in aux_Q_cnn_grads_tf]

            # Get gradient of cnn from the main task
            self.main_task_Q_cnn_grad_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q/cnn'), name='aux_update_main_gradient_Q')
            self.main_task_Q_cnn_grad_flatten_tf = flatten_grads(grads=self.main_task_Q_cnn_grad_tf,
                                                                 var_list=self._vars('main/Q/cnn'), clip_grad_range=self.clip_grad_range)

        # polyak averaging, excluding the auxiliary variables
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.main_vars = [var for var in self.main_vars if
                          var not in (self._vars('main/Q/aux') + self._vars('main/pi/aux'))]
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        assert len(self.main_vars) == len(self.target_vars)
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('bw_frame_stats') + \
                          self._global_vars('op_flow_stats') + self._global_vars('g_stats') + \
                          self._global_vars('transformed_frame_stats')

        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars)))

        tf.variables_initializer(self._global_vars('')).run()

        self._sync_optimizers()
        if self.use_aux_tasks:
            self.aux_weight_updater.sync()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        transitions = self.buffer.sample(self.batch_size)
        action_mean = np.mean(np.abs(transitions['u']))
        action_std = np.std(transitions['u'])
        logs += [('buffer_a/abs_mean', action_mean)]
        logs += [('buffer_a/std', action_std)]

        if self.use_aux_tasks:
            # Log auxiliary task losses (After the log operator)
            for (aux_task_name, aux_task_weight) in zip(self.main.name_auxiliary_tasks, self.log_aux_losses_Q):
                logs += [('aux_losses_Q/' + aux_task_name, aux_task_weight)]

            # Log auxiliary task weights
            curr_aux_weights = self.sess.run(self.aux_weight_vector_Q_tf)
            for (aux_task_name, aux_task_weight) in zip(self.main.name_auxiliary_tasks, curr_aux_weights):
                logs += [('aux_weights_Q/' + aux_task_name, aux_task_weight)]
        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', '_updater', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None
        state['serialized'] = True
        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert (len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
