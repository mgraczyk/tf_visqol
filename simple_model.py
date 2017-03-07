import tensorflow as tf
import numpy as np

from tf_visqol import TFVisqol, _DTYPE


def get_loss(ref, deg, filter_output, fs, n_samples):
  with tf.variable_scope("loss"):
    tf_visqol = TFVisqol(fs)
    before_nsim = tf_visqol.visqol(ref, deg, n_samples)
    filtered_nsim = tf_visqol.visqol(ref, filter_output, n_samples)
    nsim_loss = tf.reduce_mean(before_nsim - filtered_nsim)

    clipping_loss = 1e-1 * tf.reduce_mean(
      tf.square(tf.maximum(filter_output - 1., 0.)) + tf.square(
        tf.minimum(filter_output + 1., 0.)))

    # ref_power = tf.reduce_mean(tf.square(ref), axis=[1])
    # ref_energy = tf.sqrt(ref_power)
    # filt_energy = tf.sqrt(tf.reduce_mean(tf.square(filter_output), axis=[1]))
    # energy_loss = 1e-2 * tf.reduce_mean(
    # tf.maximum(tf.square(ref_energy - filt_energy) / ref_power - 5e-1, 0))
    energy_loss = tf.constant(0, dtype=_DTYPE)

    # sq_loss = 1e-2 * tf.log(tf.reduce_mean(tf.squared_difference(filter_output, ref)))
    # reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = nsim_loss + clipping_loss + energy_loss
    losses = {
      "loss": loss,
      "nsim": nsim_loss,
      "clipping": clipping_loss,
      "energy": energy_loss,
    }
    return losses

def get_minimize_op(loss):
  with tf.variable_scope("minimizer"):
    opt = tf.train.AdamOptimizer(learning_rate=5e-6)
    minimize_op = opt.minimize(loss)
    return minimize_op, opt

def _dense(layer_input, num_outputs):
  weights_init = tf.contrib.layers.xavier_initializer()
  layer_output = tf.contrib.layers.fully_connected(
    layer_input,
    num_outputs=num_outputs,
    weights_initializer=weights_init,
    biases_initializer=weights_init)
  return layer_output

def lrelu(x):
  return tf.maximum(0.1 * x, x)

def conv_net(deg_var,
             block_size,
             n_filters=[5, 5, 5, 5, 5, 5, 10, 40, 400],
             filter_sizes=[9, 5, 5, 5, 5, 5, 5, 5, 5],
             strides=[1, 3, 3, 3, 3, 3, 3, 3, 3]):
  assert len(filter_sizes) == len(n_filters)
  assert len(strides) == len(n_filters)

  batch_size = deg_var.get_shape().as_list()[0]
  weights_init = tf.contrib.layers.xavier_initializer_conv2d()
  conv_in = tf.expand_dims(tf.expand_dims(deg_var, axis=-1), axis=-1)

  current_input = conv_in
  outputs = []
  encoder = []
  shapes = []
  for layer_i, n_output in enumerate(n_filters):
    shape = current_input.get_shape().as_list()
    n_input = shape[3]
    shapes.append(shape)
    with tf.variable_scope("conv_in/layer_{}".format(layer_i)):
      W = tf.Variable(weights_init((filter_sizes[layer_i], 1, n_input, n_output)), name="W")
      encoder.append(W)
      output = (tf.nn.conv2d(
        current_input, W, strides=[1, strides[layer_i], 1, 1], padding='SAME'))
    outputs.append(output)
    current_input = output


  last_layer_size = current_input.get_shape().as_list()
  current_input = tf.reshape(current_input, (batch_size, -1))

  current_input = lrelu(_dense(current_input, 256))
  current_input = lrelu(_dense(current_input, 256))
  current_input = _dense(current_input, last_layer_size[1] * last_layer_size[3])
  current_input = tf.nn.tanh(current_input)
  current_input = tf.reshape(current_input, outputs[-1].get_shape())
  middle = current_input

  for layer_i, shape in reversed(list(enumerate(shapes))):
    with tf.variable_scope("conv_out/layer_{}".format(layer_i)):
      W = encoder[layer_i]
      # W = tf.Variable(
        # weights_init((filter_sizes[layer_i], 1, shape[3], current_input.get_shape()
                      # .as_list()[3])), name="W")

      b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]), name="b")
      output = (
        tf.nn.conv2d_transpose(
          current_input,
          W,
          tf.stack([tf.shape(deg_var)[0], shape[1], shape[2], shape[3]]),
          strides=[1, strides[layer_i], 1, 1],
          padding='SAME') + b)
    current_input = output

  output = tf.squeeze(current_input, [-2, -1])
  return output, middle


def get_simple_model(deg_var, block_size):
  weights_init = tf.contrib.layers.xavier_initializer()
  with tf.variable_scope("simple_model"):
    batch_size = deg_var.get_shape().as_list()[0]

    x = deg_var
    x, middle = conv_net(x, block_size)
    # scale = tf.nn.sigmoid(_dense(tf.reshape(middle, (batch_size, -1)), 1))
    # output = scale*deg_var + (1 - scale) * x
    output = 0 * deg_var + x
    return output
