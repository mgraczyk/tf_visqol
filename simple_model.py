import tensorflow as tf
import numpy as np

from tf_visqol import TFVisqol, _DTYPE


def get_loss(ref_var, deg_var, fs, n_samples):
  with tf.variable_scope("loss"):
    tf_visqol = TFVisqol(fs)
    nsim = tf_visqol.visqol(ref_var, deg_var, n_samples)
    sq_loss = tf.cast(tf.reduce_mean(tf.squared_difference(deg_var, ref_var)), _DTYPE)
    loss = 0.1*tf.log(sq_loss) - tf.reduce_mean(nsim)
    return loss

def get_minimize_op(loss):
  with tf.variable_scope("minimizer"):
    minimize_op = tf.train.AdamOptimizer().minimize(loss)
    return minimize_op

def _dense(layer_input, num_outputs):
  block_size = tf.shape(layer_input)[1]
  weights_init = tf.contrib.layers.xavier_initializer()
  layer_output = tf.contrib.layers.fully_connected(
    layer_input,
    num_outputs=num_outputs,
    weights_initializer=weights_init,
    biases_initializer=weights_init)
  return layer_output


def get_simple_model(deg_var, block_size):
  weights_init = tf.contrib.layers.xavier_initializer()
  with tf.variable_scope("simple_model"):
    x = deg_var

    x = _dense(x, block_size // 16)
    x = tf.nn.elu(x)

    x = _dense(x, block_size)
    x = tf.nn.tanh(x)

    output = deg_var + x
    return output
