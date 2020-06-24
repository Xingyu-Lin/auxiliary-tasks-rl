# Created by Xingyu Lin, 25/03/2018
import tensorflow as tf
from olaux.utils import store_args, nn
import numpy as np

cnn_num_filters = 64


def cnn_one_stream(input_net, dim_latent_repr, cnn_nonlinear, use_bottleneck_layer, scope='cnn', reuse=False, return_shape=False):
    act_fn = tf.nn.tanh if cnn_nonlinear == 'tanh' else tf.nn.relu
    with tf.variable_scope(scope, reuse=reuse):
        net = tf.layers.conv2d(name='conv1', inputs=input_net, filters=cnn_num_filters//4, strides=2, kernel_size=[2, 2],
                               padding='same', activation=act_fn)
        net = tf.layers.conv2d(name='conv2', inputs=net, filters=cnn_num_filters//2, strides=2, kernel_size=[2, 2],
                               padding='same', activation=act_fn)
        net = tf.layers.conv2d(name='conv3', inputs=net, filters=cnn_num_filters, strides=2, kernel_size=[2, 2], padding='valid',
                               activation=act_fn)
        net = tf.layers.conv2d(name='conv4', inputs=net, filters=cnn_num_filters, strides=2, kernel_size=[2, 2],
                               padding='valid', activation=act_fn)
        shape = net.get_shape().as_list()  # a list: [None, 7, 7, 64]
        dim = np.prod(shape[1:])
        net = tf.reshape(net, [-1, dim])
        if use_bottleneck_layer:
            net = tf.layers.dense(name='conv_fc', inputs=net, units=dim_latent_repr)
    return (net, shape) if return_shape else net


def cnn_upstream(input_net, conv_shape, use_bottleneck_layer, cnn_nonlinear, scope='cnn', reuse=False, final_layer_depth=1):
    act_fn = tf.nn.tanh if cnn_nonlinear == 'tanh' else tf.nn.relu

    with tf.variable_scope(scope, reuse=reuse):
        if use_bottleneck_layer:
            dim_conv = np.prod(conv_shape[1:])
            net = tf.layers.dense(name='fc_conv', inputs=input_net, units=dim_conv)
            net = tf.reshape(net, [-1] + conv_shape[1:])
        else:
            if scope == 'opflow':
                conv_shape = conv_shape.copy()
                conv_shape[-1] *= 2  # Two conv features are stacked
                net = tf.reshape(input_net, [-1] + conv_shape[1:])
            else:
                net = tf.reshape(input_net, [-1] + conv_shape[1:])

        net = tf.layers.conv2d_transpose(name='deconv1', inputs=net, filters=cnn_num_filters,
                                         kernel_size=[2, 2], strides=2, activation=act_fn, padding='same')
        net = tf.layers.conv2d_transpose(name='deconv2', inputs=net, filters=cnn_num_filters,
                                         kernel_size=[3, 3], padding='valid', strides=2, activation=act_fn)
        net = tf.layers.conv2d_transpose(name='deconv3', inputs=net, filters=cnn_num_filters//4, kernel_size=[2, 2],
                                         padding='same', activation=act_fn, strides=2)
        net = tf.layers.conv2d_transpose(name='deconv4', inputs=net, filters=final_layer_depth, kernel_size=[2, 2],
                                         padding='same', strides=2)
    return net


# noinspection PyAttributeOutsideInit,PyPep8Naming,SpellCheckingInspection
class CNNActorCritic:
    @store_args
    def __init__(self, inputs_tf, image_input_shapes, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 dim_latent_repr, cnn_nonlinear, use_bottleneck_layer, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        self.inputs_tf = inputs_tf
        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        self.image_shape = image_input_shapes['o']
        o = tf.reshape(o, [-1, *self.image_shape])
        g = tf.reshape(g, [-1, *self.image_shape])

        # Networks.
        self.x_o_pi = cnn_one_stream(o, self.dim_latent_repr, cnn_nonlinear, use_bottleneck_layer, scope='pi/cnn', reuse=False)
        self.x_o_q = cnn_one_stream(o, self.dim_latent_repr, cnn_nonlinear, use_bottleneck_layer, scope='Q/cnn', reuse=False)

        x_g_pi = cnn_one_stream(g, self.dim_latent_repr, cnn_nonlinear, use_bottleneck_layer, scope='pi/cnn', reuse=True)
        x_g_q = cnn_one_stream(g, self.dim_latent_repr, cnn_nonlinear, use_bottleneck_layer, scope='Q/cnn', reuse=True)

        with tf.variable_scope('pi/fc'):
            x_concat = tf.concat(axis=1, values=[self.x_o_pi, x_g_pi])
            self.pi_tf_pre = nn(x_concat, [self.hidden] * self.layers + [self.dimu], name='mlp')
            self.pi_tf = self.max_u * tf.nn.tanh(self.pi_tf_pre)

        with tf.variable_scope('Q/fc'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[self.x_o_q, x_g_q, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])

            # for critic training
            input_Q = tf.concat(axis=1, values=[self.x_o_q, x_g_q, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

    def build_auxiliary_tasks(self):
        # Prepare auxiliary info used for training auxiliary tasks
        self.action_taken = self.u_tf
        self.transformation = self.inputs_tf['info_transformation']

        next_frame = self.o_stats.normalize(self.inputs_tf['o_2'])
        next_frame = tf.reshape(next_frame, [-1, *self.image_shape])

        transformed_frame = self.transformed_frame_stats.normalize(self.inputs_tf['info_transformed_frame'])
        self.transformed_frame = tf.reshape(transformed_frame, [-1, *self.image_shape])

        self.op_flow = self.op_flow_stats.normalize(self.inputs_tf['info_op_flow'])
        self.op_flow = tf.reshape(self.op_flow, [-1, *self.image_shape[:2], 1])

        self.bw_frame = self.bw_frame_stats.normalize(self.inputs_tf['info_bw_frame'])  # black and white reconstruction
        self.bw_frame = tf.reshape(self.bw_frame, [-1, *self.image_shape[:2], 1])

        # Get representation used for auxiliary tasks
        self.x_prev_frame_Q = self.x_o_q
        self.x_next_frame_Q = cnn_one_stream(next_frame, self.dim_latent_repr, self.cnn_nonlinear, self.use_bottleneck_layer, scope='Q/cnn',
                                             reuse=True)
        self.x_transformed_frame_Q = cnn_one_stream(self.transformed_frame, self.dim_latent_repr, self.cnn_nonlinear, self.use_bottleneck_layer,
                                                    scope='Q/cnn',
                                                    reuse=True)

        self.x_prev_frame_pi = self.x_o_pi
        self.x_next_frame_pi, conv_shape = cnn_one_stream(next_frame, self.dim_latent_repr, self.cnn_nonlinear, self.use_bottleneck_layer,
                                                          scope='pi/cnn', reuse=True,
                                                          return_shape=True)
        self.x_transformed_frame_pi = cnn_one_stream(self.transformed_frame, self.dim_latent_repr, self.cnn_nonlinear, self.use_bottleneck_layer,
                                                     scope='pi/cnn',
                                                     reuse=True)

        # Add auxiliary task networks for the critic
        dim_action_prediction = int(self.action_taken.shape[1])
        dim_transformation_prediction = int(self.transformation.shape[1])
        dim_latent_repr_prediction = int(np.prod(self.x_prev_frame_Q.get_shape().as_list()[1:]))
        with tf.variable_scope('Q/aux'):  # Use aux sub-scope so that these can be excluded from the target net
            x_neighbouring_frame_Q = tf.concat(axis=1, values=[self.x_prev_frame_Q, self.x_next_frame_Q])
            self.aux_predicted_op_flow_Q = cnn_upstream(x_neighbouring_frame_Q, conv_shape, self.use_bottleneck_layer, self.cnn_nonlinear,
                                                        scope='opflow', reuse=False)

            self.aux_predicted_action_Q = tf.layers.dense(name='aux_predict_action', inputs=x_neighbouring_frame_Q,
                                                          units=dim_action_prediction)
            x_neighbouring_transformed_frame_Q = tf.concat(axis=1, values=[self.x_prev_frame_Q,
                                                                           self.x_transformed_frame_Q])
            self.aux_predicted_transformation_Q = tf.layers.dense(name='aux_predict_transformation',
                                                                  inputs=x_neighbouring_transformed_frame_Q,
                                                                  units=dim_transformation_prediction)
            x_current_repr_action = tf.concat(axis=1, values=[self.x_prev_frame_Q, self.action_taken])
            self.aux_predicted_latent_repr_Q = tf.layers.dense(name='aux_predict_latent_repr',
                                                               inputs=x_current_repr_action,
                                                               units=dim_latent_repr_prediction)
            self.aux_ae_reconstr_Q = cnn_upstream(self.x_prev_frame_Q, conv_shape, self.use_bottleneck_layer, self.cnn_nonlinear,
                                                  scope='autoencoder', reuse=False, final_layer_depth=1)

        # Construct auxiliary task losses for the critic
        self.loss_op_flow_Q_tf = tf.reduce_mean(tf.square(self.aux_predicted_op_flow_Q - self.op_flow))
        self.loss_action_prediction_Q_tf = tf.reduce_mean(tf.square(self.aux_predicted_action_Q - self.action_taken))
        self.loss_egomotion_Q_tf = tf.reduce_mean(tf.square(self.aux_predicted_transformation_Q - self.transformation))
        self.loss_latent_repr_prediction_Q_tf = tf.reduce_mean(tf.square(self.aux_predicted_latent_repr_Q - self.x_next_frame_Q))
        self.loss_autoencoder_Q_tf = tf.reduce_mean(tf.square(self.aux_ae_reconstr_Q - self.bw_frame))

        self.loss_auxiliary_tasks_Q_tf = [self.loss_action_prediction_Q_tf,
                                          self.loss_latent_repr_prediction_Q_tf,
                                          self.loss_egomotion_Q_tf,
                                          self.loss_op_flow_Q_tf,
                                          self.loss_autoencoder_Q_tf]

        # Add auxiliary task networks for the policy
        with tf.variable_scope('pi/aux'):  # Use aux sub-scope so that these can be excluded from the target net
            x_neighbouring_frame_pi = tf.concat(axis=1, values=[self.x_prev_frame_pi, self.x_next_frame_pi])
            self.aux_predicted_op_flow_pi = cnn_upstream(x_neighbouring_frame_pi, conv_shape, self.use_bottleneck_layer, self.cnn_nonlinear,
                                                         scope='opflow', reuse=False)
            self.aux_predicted_action_pi = tf.layers.dense(name='aux_predict_action', inputs=x_neighbouring_frame_pi,
                                                           units=dim_action_prediction)
            x_neighbouring_transformed_frame_pi = tf.concat(axis=1, values=[self.x_prev_frame_pi,
                                                                            self.x_transformed_frame_pi])
            self.aux_predicted_transformation_pi = tf.layers.dense(name='aux_predict_transformation',
                                                                   inputs=x_neighbouring_transformed_frame_pi,
                                                                   units=dim_transformation_prediction)
            x_current_repr_action = tf.concat(axis=1, values=[self.x_prev_frame_pi, self.action_taken])
            self.aux_predicted_latent_repr_pi = tf.layers.dense(name='aux_predict_latent_repr',
                                                                inputs=x_current_repr_action,
                                                                units=dim_latent_repr_prediction)
            self.aux_ae_reconstr_pi = cnn_upstream(self.x_prev_frame_pi, conv_shape, self.use_bottleneck_layer, self.cnn_nonlinear,
                                                   scope='autoencoder', reuse=False, final_layer_depth=1)

        # Construct auxiliary task losses for the policy
        self.loss_op_flow_pi_tf = tf.reduce_mean(tf.square(self.aux_predicted_op_flow_pi - self.op_flow))
        self.loss_action_prediction_pi_tf = tf.reduce_mean(tf.square(self.aux_predicted_action_pi - self.action_taken))
        self.loss_egomotion_pi_tf = tf.reduce_mean(tf.square(self.aux_predicted_transformation_pi - self.transformation))
        self.loss_latent_repr_prediction_pi_tf = tf.reduce_mean(tf.square(self.aux_predicted_latent_repr_pi - self.x_next_frame_pi))
        self.loss_autoencoder_pi_tf = tf.reduce_mean(tf.square(self.aux_ae_reconstr_pi - self.bw_frame))
        self.loss_auxiliary_tasks_pi_tf = [self.loss_action_prediction_pi_tf,
                                           self.loss_latent_repr_prediction_pi_tf,
                                           self.loss_egomotion_pi_tf,
                                           self.loss_op_flow_pi_tf,
                                           self.loss_autoencoder_pi_tf]
        self.name_auxiliary_tasks = ['inverse_dynamics', 'forward_dynamics', 'egomotion', 'optical_flow', 'autoencoder']
