import tensorflow as tf
import numpy as np

from tf_visqol import TFVisqol, _DTYPE


def get_loss(ref, filter_output, fs, n_samples):
  with tf.variable_scope("loss"):
    tf_visqol = TFVisqol(fs)
    nsim = tf_visqol.visqol(ref, filter_output, n_samples)
    sq_loss = tf.cast(tf.reduce_mean(tf.squared_difference(filter_output, ref)), _DTYPE)
    loss = 1e-3*tf.log(sq_loss) - tf.reduce_mean(nsim)
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

def conv_net(deg_var, block_size,
                n_filters=[10],
                filter_sizes=[3],
                dropout=False):
  weights_init = tf.contrib.layers.xavier_initializer()
  conv_in = tf.expand_dims(tf.expand_dims(deg_var, axis=-1), axis=-1)
  strides = [1, 1, 1, 1]

  current_input = conv_in
  encoder = []
  shapes = []
  for layer_i, n_output in enumerate(n_filters):
    n_input = current_input.get_shape().as_list()[3]
    shapes.append(current_input.get_shape().as_list())
    W = tf.Variable(weights_init((filter_sizes[layer_i], 1, n_input, n_output)))
    b = tf.Variable(tf.zeros([n_output]))
    encoder.append(W)
    output = tf.nn.elu(
        tf.add(tf.nn.conv2d(
            current_input, W, strides=strides, padding='SAME'), b))
    current_input = output

  if dropout:
    current_input = tf.nn.dropout(current_input, 0.5)

  encoder.reverse()
  shapes.reverse()

  for layer_i, shape in enumerate(shapes):
    W = encoder[layer_i]
    b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
    output = tf.nn.elu(tf.add(
        tf.nn.conv2d_transpose(
            current_input, W,
            tf.stack([tf.shape(deg_var)[0], shape[1], shape[2], shape[3]]),
            strides=strides, padding='SAME'), b))
    current_input = output

  output = tf.squeeze(current_input, [-2, -1])
  return output


def get_simple_model(deg_var, block_size):
  weights_init = tf.contrib.layers.xavier_initializer()
  with tf.variable_scope("simple_model"):
    x = deg_var


    # x = _dense(x, block_size // 16)
    # x = tf.nn.elu(x)
    # x = tf.nn.dropout(x, 0.8)

    # x = _dense(x, block_size)
    # x = tf.nn.tanh(x)

    x = conv_net(x, block_size)
    output = deg_var + x
    return output
