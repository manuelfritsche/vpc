from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import constants


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def universeHead(x, nConvs=4):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    print('Using universe head design')
    for i in range(nConvs):
        x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # print('Loop{} '.format(i+1),tf.shape(x))
        # print('Loop{}'.format(i+1),x.get_shape())
    x = flatten(x)
    return x

def LSTMcell(x, step_size, size, c_in, h_in):
    # introduce a "fake" batch dimension of 1 to do LSTM over time dim
    x = tf.expand_dims(x, [0])
    lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)

    
    
    ###<- self.state_in = [c_in, h_in]

    state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm, x, initial_state=state_in, sequence_length=step_size,
        time_major=False)
    return lstm.state_size, lstm_outputs, lstm_state


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, use_pred=False):
        self.use_pred = use_pred
        
        # A3C model
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        size = 256
        x = universeHead(x)
        
        # LSTM a3c
        # introduce a "fake" batch dimension of 1 to do LSTM over time dim
        x = tf.expand_dims(x, [0])
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        
        x = tf.reshape(lstm_outputs, [-1, size])
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # STATE PREDICTOR -----------------------------------------------------
        if use_pred:
            # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
            # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
            input_shape = [None] + list(ob_space)
            self.s1 = phi1 = tf.placeholder_with_default(np.zeros(shape=[1] + list(ob_space), dtype=np.float32), input_shape)
            self.s2 = phi2 = tf.placeholder_with_default(np.zeros(shape=[1] + list(ob_space), dtype=np.float32), input_shape)
            self.asample = asample = tf.placeholder_with_default(np.zeros([1, ac_space], dtype=np.float32), [None, ac_space])
    
            # feature encoding: phi1, phi2: [None, LEN]
            size = 256   
            # use same head as for value function
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi1 = universeHead(phi1)
                phi2 = universeHead(phi2)
    
            # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
            g = tf.concat(1,[phi1, phi2])
            g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
            aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
            logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
            self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits, aindex), name="invloss")
            self.ainvprobs = tf.nn.softmax(logits, dim=-1)
    
            # forward model: f(phi1,asample) -> phi2
            # Note: no backprop to asample of policy: it is treated as fixed for predictor training
            f = tf.concat(1, [phi1, asample])
            f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
            f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
            self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
            self.forwardloss = self.forwardloss * 288.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.
            self.pred_state = f
        
            # LSTM - VALUE PREDICTION --------------------------------------------- 
            x_pred = tf.expand_dims(f, [0])
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                lstm_outputs, _ = tf.nn.dynamic_rnn(
                    lstm, x_pred, initial_state=state_in, sequence_length=step_size,
                    time_major=False)
                x_pred = tf.reshape(lstm_outputs, [-1, size])
                self.vf_pred = tf.reshape(linear(x_pred, 1, "value"), [-1])
          
         
        # [0, :] means pick action of first state from batch. Hardcoded b/c
        # batch=1 during rollout collection. Its not used during batch training.
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # tf.add_to_collection('probs', self.probs)
        # tf.add_to_collection('sample', self.sample)
        # tf.add_to_collection('state_out_0', self.state_out[0])
        # tf.add_to_collection('state_out_1', self.state_out[1])
        # tf.add_to_collection('vf', self.vf)

    def get_initial_features(self):
        # Call this function to get reseted lstm memory cells
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

    def pred_value(self, s1, asample, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf_pred, {self.s1: [s1], self.asample: [asample], self.state_in[0]: c, self.state_in[1]: h})[0]
        

    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        # error = sess.run([self.forwardloss, self.invloss],
        #     {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error[0], ' ErrorI:', error[1])
        error = sess.run(self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        error = error * constants['PREDICTION_BETA'] * self.use_pred
        return error