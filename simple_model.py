import tensorflow as tf
import numpy as np

from tf_visqol import TFVisqol, _DTYPE

from tf_util import define_scope

def get_loss(ref, deg, filter_output, fs, n_samples):
  with tf.variable_scope("loss"):
    tf_visqol = TFVisqol(fs)
    before_nsim = tf_visqol.visqol(ref, deg, n_samples)
    filtered_nsim = tf_visqol.visqol(ref, filter_output, n_samples)
    nsim_loss = tf.reduce_mean(before_nsim - filtered_nsim)

    clipping_loss = 1e-2 * tf.reduce_mean(
      tf.maximum(filter_output - 1., 0.) - tf.minimum(filter_output + 1., 0.))

    ref_power = tf.reduce_mean(tf.square(ref), axis=[1])
    ref_energy = tf.sqrt(1e-4 + ref_power)
    filt_energy = tf.sqrt(1e-4 + tf.reduce_mean(tf.square(filter_output), axis=[1]))
    energy_loss = 1e-1 * tf.reduce_mean(tf.maximum(tf.square(ref_energy - filt_energy) / ref_power - 5e-1, 0))
    # energy_loss = tf.constant(0, dtype=_DTYPE)

    # sq_loss = 1e-2 * tf.log(tf.reduce_mean(tf.squared_difference(filter_output, ref)))
    # reg_loss = tf.reduce_mean(tf.losses.get_regularization_losses())
    reg_loss = tf.constant(0, dtype=_DTYPE)

    losses = {
      "nsim": nsim_loss,
      "clipping": clipping_loss,
      "energy": energy_loss,
      "reg": reg_loss
    }
    losses["loss"] = sum(losses.values())
    return losses

def get_minimize_op(loss):
  with tf.variable_scope("minimizer"):
    opt = tf.train.AdamOptimizer(learning_rate=3e-6)
    minimize_op = opt.minimize(loss)
    return minimize_op, opt

def _dense(layer_input, num_outputs):
  weights_init = tf.contrib.layers.xavier_initializer()
  layer_output = tf.contrib.layers.fully_connected(
    layer_input,
    num_outputs=num_outputs,
    weights_initializer=weights_init,
    biases_initializer=tf.constant_initializer(0.))
  return layer_output

def lrelu(x):
  return tf.maximum(0.1 * x, x)

def conv_net(deg_var,
             n_filters=[10, 8, 8, 8, 8, 8, 10, 40, 70],
             filter_sizes=[3, 5, 5, 5, 5, 5, 5, 5, 5],
             strides=[1, 2, 3, 3, 3, 3, 3, 3, 3]):
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

  current_input = tf.nn.elu(_dense(current_input, 256))
  current_input = tf.nn.elu(_dense(current_input, 256))
  current_input = _dense(current_input, last_layer_size[1] * last_layer_size[3])
  current_input = tf.nn.tanh(current_input)
  current_input = tf.reshape(current_input, outputs[-1].get_shape())
  middle = current_input

  for layer_i, shape in reversed(list(enumerate(shapes))):
    with tf.variable_scope("conv_out/layer_{}".format(layer_i)):
      #  Don't weight tie
      W = tf.Variable(
        weights_init((filter_sizes[layer_i], 1, shape[3], current_input.get_shape()
                      .as_list()[3])), name="W")

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
    x, middle = conv_net(x)
    # scale = tf.nn.sigmoid(_dense(tf.reshape(middle, (batch_size, -1)), 1))

    # Try to drive scale to zero.
    # tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1e-2), [scale])

    # output = scale*deg_var + (1 - scale) * x
    output = 0 * deg_var + x

    # Regularize the output so that we learn to be silent when the output is silent.
    # tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1e-3),
    # [tf.reduce_mean(tf.abs(output), axis=[1])])
    return output

def assertShape(var, shape):
  varshape = var.get_shape()
  assert varshape == shape, "{} != {}".format(varshape, shape)

def get_preprocessed(x, n_filters_pre):
  weights_init = tf.contrib.layers.xavier_initializer()
  batch_size = x.get_shape()[0]
  block_size = x.get_shape()[1]

  # Compute a representation of the input.
  filter_size_pre = 16
  with tf.variable_scope("proc_conv"):
    W = tf.Variable(weights_init((filter_size_pre, 1, n_filters_pre)), name="W")
    b = tf.Variable(tf.zeros([n_filters_pre]), name="b")
    x = tf.nn.conv1d(x, W, 1, "SAME")
    x = tf.nn.bias_add(x, b)
  preprocessed = x
  assertShape(preprocessed, (batch_size, block_size, n_filters_pre))
  return preprocessed

def get_attention(representation, n_filters_pre):
  weights_init = tf.contrib.layers.xavier_initializer()
  batch_size = representation.get_shape()[0]
  block_size = representation.get_shape()[1]

  # Filter the input selectively based on the representation.
  filter_size_attn = 16
  n_filters_attn = 10
  x = representation
  with tf.variable_scope("filter_conv"):
    W = tf.Variable(weights_init((filter_size_attn, 1, n_filters_attn)), name="W")
    b = tf.Variable(tf.zeros([n_filters_attn]), name="b")
    x = tf.nn.conv1d(x, W, 1, "SAME")
    x = tf.nn.bias_add(x, b)
  preattention = x
  assertShape(preattention, (batch_size, block_size, n_filters_attn))

  x = preattention
  with tf.variable_scope("filter_attention"):
    W = tf.Variable(weights_init((n_filters_attn, n_filters_pre)), name="W")
    b = tf.Variable(tf.zeros([n_filters_pre]), name="b")
    x = tf.tensordot(x, W, axes=((2,), (0,)))
    x = tf.nn.bias_add(x, b)
    x = tf.nn.elu(x)
    x = tf.nn.softmax(x)
  filter_attention = x
  assertShape(filter_attention, (batch_size, block_size, n_filters_pre))

  return filter_attention

@define_scope
def get_baseline_model(deg_var, block_size):
  n_filters_pre = 20
  batch_size = deg_var.get_shape().as_list()[0]
  assertShape(deg_var, (batch_size, block_size))

  deg_channels = tf.expand_dims(deg_var, axis=-1)
  assertShape(deg_channels, (batch_size, block_size, 1))

  with tf.variable_scope("deg_preprocess"):
    preprocessed = get_preprocessed(deg_channels, n_filters_pre)

  with tf.variable_scope("attention"):
    filter_attention = get_attention(deg_channels, n_filters_pre)

  with tf.variable_scope("apply_attention"):
    filtered_signal = tf.einsum("ijk,ijk->ij", preprocessed, filter_attention)
  assertShape(filtered_signal, (batch_size, block_size))

  # Scale output so it is in the same range.
  output = 10*filtered_signal
  return output

@define_scope
def get_noise_filling_model(deg_var, block_size):
  n_filters_pre = 20
  batch_size = deg_var.get_shape().as_list()[0]
  assertShape(deg_var, (batch_size, block_size))

  deg_channels = deg_var[..., None]
  assertShape(deg_channels, (batch_size, block_size, 1))

  with tf.variable_scope("deg_preprocess"):
    preprocessed = get_preprocessed(deg_channels, n_filters_pre)

  with tf.variable_scope("deg_attention"):
    filter_attention = get_attention(deg_channels, n_filters_pre)

  with tf.variable_scope("apply_deg_attention"):
    filtered_signal = tf.einsum("ijk,ijk->ij", preprocessed, filter_attention)
  assertShape(filtered_signal, (batch_size, block_size))

  signal_mean, signal_power = tf.nn.moments(deg_var, axes=[1])
  noise = 0.1 * (tf.sqrt(signal_power[:, None]) * tf.random_normal(
    tf.shape(deg_var), mean=0.0, stddev=1.0))[..., None]

  noise_filter = tf.constant(np.ones((2, 1, 1), dtype=np.float32) / 2., dtype=_DTYPE)
  noise = tf.nn.conv1d(noise, noise_filter, 1, "SAME")
  with tf.variable_scope("noise_preprocess"):
    noise_preprocessed = get_preprocessed(noise, n_filters_pre)

  with tf.variable_scope("noise_attention"):
    noise_attention = get_attention(deg_channels, n_filters_pre)

  with tf.variable_scope("apply_noise_attention"):
    filtered_noise = tf.einsum("ijk,ijk->ij", noise_preprocessed, noise_attention)
  assertShape(filtered_noise, (batch_size, block_size))

  # Scale output so it is in the same range.
  output = 10*(filtered_signal + filtered_noise)

  return output
