import numpy as np
import tensorflow as tf
import functools


_DTYPE = tf.float32


_BFS_ARR = np.asarray([
    50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900,
    3400, 4000, 4800, 6500, 8000
  ])
_BFS = tf.constant(_BFS_ARR, dtype=_DTYPE)

_NUM_BANDS = len(_BFS_ARR)
_PATCH_SIZE = 30
_PI = np.pi
_BLOCK_SIZE = 512


def define_scope(function, scope=None, *args, **kwargs):
  """args and kwargs are passed to variable_scope."""
  scope = scope or function.__name__
  vs_args = args
  vs_kwargs = kwargs
  @functools.wraps(function)
  def decorator(*args, **kwargs):
    with tf.variable_scope(scope, *vs_args, **vs_kwargs):
      return function(*args, **kwargs)
  return decorator


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

# Adapted from
#   http://www.mathworks.com/matlabcentral/fileexchange/35103-generalized-goertzel-algorithm/content/goertzel_general_shortened.m
@define_scope
def gga_freq_abs(x, sample_rate, freq):
  """Computes the magnitude of the time domain signal x at each frequency in freq using
     the generalized Goertzel algorithm.

     x has shape (batch, block)
  """
  # TODO: This is slow. Any way to improve it?
  lx = _BLOCK_SIZE
  pik_term = 2 * _PI * freq / sample_rate
  cos_pik_term = tf.cos(pik_term)
  cos_pik_term2 = 2 * cos_pik_term

  # TODO: Maybe if we make these states into proper variables and assign to them,
  #       we will use less memory.
  # Use tf.zeros because zeros_initializer doesn't seem to work in tf 1.0.
  shape = (x.get_shape()[0], _NUM_BANDS)
  s0 = tf.Variable(tf.zeros(shape, dtype=_DTYPE), trainable=False)
  s1 = tf.Variable(tf.zeros(shape, dtype=_DTYPE), trainable=False)
  s2 = tf.Variable(tf.zeros(shape, dtype=_DTYPE), trainable=False)

  # number of iterations is (by one) less than the length of signal
  # Pipeline the first two iterations.
  s1 = tf.assign(s1, tf.tile(x[:, 0, None], (1, _NUM_BANDS)))
  s0 = tf.assign(s0, x[:, 1, None] + cos_pik_term2 * s1)
  s2 = tf.assign(s2, s1)
  s1 = tf.assign(s1, s0)

  for ind in range(2, lx - 1):
    s0 = tf.assign(s0, x[:, ind, None] + cos_pik_term2 * s1 - s2)

    # We have to tell tensorflow explicitly that the above assignment to s0 happens before the
    # following assignments.
    with tf.control_dependencies([s0]):
      s2 = tf.assign(s2, s1)
    with tf.control_dependencies([s2]):
      s1 = tf.assign(s1, s0)

  # s0 = tf.assign(s0, x[:, lx - 1, None] + cos_pik_term2 * s1 - s2)
  s0 = x[:, lx - 1, None] + cos_pik_term2 * s1 - s2

  # | s0 - s1 exp(-ip) |
  # | s0 - s1 cos(p) + i s1 sin(p)) |
  # sqrt((s0 - s1 cos(p))^2 + (s1 sin(p))^2)
  # sqrt(s0^2 - 2 s0 s1 cos(p) + s1^2 cos^2(p) + s1^2 sin^2(p))
  # sqrt(s0^2 - + s1^2 - 2 s0 s1 cos(p))

  # TODO: Figure out why this doesn't work.
  # y = tf.sqrt(tf.square(s0) + tf.square(s1) - (s0*s1)*cos_pik_term2)
  y = tf.sqrt((s0 - s1*cos_pik_term)**2 + (s1 * tf.sin(pik_term))**2)
  return y


@define_scope
def spectrogram_abs(x, window, window_overlap, bfs, fs):
  # TODO: We may need to pad for the last block.
  x_as_image = tf.expand_dims(tf.expand_dims(x, 1, name="expand_spec"), -1)
  x_as_image = tf.cast(x_as_image, tf.float32)
  blocks_raw = tf.extract_image_patches(
    x_as_image,
    ksizes=[1, 1, _BLOCK_SIZE, 1],
    strides=[1, 1, window_overlap, 1],
    rates=[1, 1, 1, 1],
    padding="VALID")
  blocks = tf.squeeze(blocks_raw, [1])
  blocks = tf.cast(blocks, _DTYPE)
  windows_blocks = window * blocks

  wb_flat = tf.reshape(windows_blocks, (-1, _BLOCK_SIZE))
  S_flat = gga_freq_abs(wb_flat, fs, bfs)
  S = tf.transpose(tf.reshape(S_flat, (tf.shape(x)[0], -1, _NUM_BANDS)), perm=(0, 2, 1))

  return S

@define_scope
def filter2(h, X, shape):
  # The MATLAB version truncates the border.
  shape = shape.upper()
  assert shape == "VALID"

  # The original performs correlation, this is convolution.
  # The difference doesn't matter because the filter is rotationally symmetric.

  X = tf.expand_dims(X, -1, name="expand_filter")

  # TODO Tensorflow doesn't support 64-bit conv.
  result = tf.nn.conv2d(tf.cast(X, tf.float32), tf.cast(h, tf.float32), strides=[1, 1, 1, 1], padding=shape)
  result = tf.squeeze(result, [-1])
  result = tf.cast(result, _DTYPE)
  return result

@define_scope
def nsim(neuro_r, neuro_d, L):
  # neuro_r and neuro_d are Tensors of (batch, freq, patch)
  window = np.array([[0.0113, 0.0838, 0.0113],
                     [0.0838, 0.6193, 0.0838],
                     [0.0113, 0.0838, 0.0113]]).reshape(3, 3, 1, 1)
  window = window / np.sum(window)
  window = tf.constant(window, dtype=_DTYPE)

  K1 = 0.01
  K2 = 0.03
  C1 = (K1 * L)**2
  C2 = ((K2 * L)**2) / 2

  # MATLAB uses double precision, but we can't because conv2d doesn't support it.
  mu_r = filter2(window, neuro_r, 'valid')
  mu_d = filter2(window, neuro_d, 'valid')
  mu_r_sq = tf.square(mu_r)
  mu_d_sq = tf.square(mu_d)
  mu_r_mu_d = mu_r * mu_d
  sigma_r_sq = filter2(window, neuro_r * neuro_r, 'valid') - mu_r_sq
  sigma_d_sq = filter2(window, neuro_d * neuro_d, 'valid') - mu_d_sq
  sigma_r_d = filter2(window, neuro_r * neuro_d, 'valid') - mu_r_mu_d
  sigma_r = tf.sign(sigma_r_sq) * tf.sqrt(tf.abs(sigma_r_sq))
  sigma_d = tf.sign(sigma_d_sq) * tf.sqrt(tf.abs(sigma_d_sq))
  L_r_d = (2. * mu_r * mu_d + C1) / (mu_r_sq + mu_d_sq + C1)
  S_r_d = (sigma_r_d + C2) / (sigma_r * sigma_d + C2)

  # Why is this here?
  nmap = tf.sign(L_r_d) * tf.abs(L_r_d) * tf.sign(S_r_d) * tf.abs(S_r_d)

  mNSIM = tf.reduce_mean(nmap, axis=[1, 2])
  return mNSIM

class TFVisqol(object):
  def __init__(self, fs):
    self._fs = fs
    if self._fs != 16000:
      raise NotImplementedError

    window_size = _BLOCK_SIZE
    assert window_size == round((self._fs / 8000) * (_BLOCK_SIZE/2))
    window_size = 2 * (window_size // 2)

    self._window_overlap = int(window_size / 2)
    self._window = tf.constant(np.hamming(window_size + 1)[:window_size], dtype=_DTYPE)

  def visqol_with_session(self, ref, deg):
    with tf.Session() as sess:
      ref_var = tf.placeholder(_DTYPE, ref.shape, name="ref")
      deg_var = tf.placeholder(_DTYPE, deg.shape, name="deg")
      nsim_var = self.visqol(ref_var, deg_var, ref.shape[1])

      # Initialize state.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      feed_dict = {ref_var: ref, deg_var: deg}
      nsim = sess.run(nsim_var, feed_dict)
      return nsim

  def visqol(self, ref_var, deg_var, n_samples):
    # TODO HACK: We pass n_samples here because of a problem with the image patch gradient.
    # TODO: How are we supposed to specify a variable that may or may not receive a feed?
    with tf.variable_scope("visqol"):
      nsim_var = self._visqol_op(ref_var, deg_var, n_samples)
      return nsim_var

  def _visqol_op(self, ref, deg, n_samples):
    tf.assert_equal(tf.shape(ref), tf.shape(deg))

    num_blocks = (n_samples // self._window_overlap) - 1
    img_rsig = tf.identity(self._get_sig_spect(ref, num_blocks), name="img_rsig")
    img_dsig = tf.identity(self._get_sig_spect(deg, num_blocks), name="img_dsig")

    lowfloor = tf.reduce_min(img_rsig)
    img_rsig = img_rsig - lowfloor
    img_dsig = img_dsig - lowfloor
    L = 160

    ref_patches = tf.identity(self.create_patches(img_rsig), name="ref_patches")
    deg_patches = tf.identity(self.create_patches(img_dsig), name="deg_patches")
    nsim = self.calc_patch_similarity(ref_patches, deg_patches, L)

    return nsim

  def _get_sig_spect(self, x, num_blocks):
    S = spectrogram_abs(x, self._window, self._window_overlap, _BFS, self._fs)

    # TODO HACK: This reshape is here because extract_image_patches gradient seems to have a bug.
    #   http://stackoverflow.com/questions/41841713/tensorflow-gradient-unsupported-operand-type
    S = tf.reshape(S, (tf.shape(x)[0], _NUM_BANDS, num_blocks))

    S = tf.maximum(S, tf.constant(1e-20, dtype=_DTYPE))
    max_S = tf.reduce_max(S)
    S /= max_S
    spec_bf = 20*log10(S)
    return spec_bf

  def create_patches(self, img_sig):
    # TODO: This slice is done in the MATLAB, but seems dumb.
    original_num_blocks = tf.shape(img_sig)[2]
    begin = int(_PATCH_SIZE / 2) - 1
    img_rsig_trunc = tf.slice(img_sig, begin=[0, 0, begin], size=[-1, -1, -1])
    img_4d = tf.expand_dims(img_rsig_trunc, -1, name="expand_patches")
    img_4d = tf.cast(img_4d, dtype=tf.float32)
    patches = tf.extract_image_patches(
      img_4d,
      ksizes=[1, 1, _PATCH_SIZE, 1],
      strides=[1, 1, _PATCH_SIZE, 1],
      rates=[1, 1, 1, 1],
      padding="VALID")
    patches = tf.transpose(patches, perm=[0, 2, 1, 3])
    patches = tf.cast(patches, _DTYPE)

    return patches


  def calc_patch_similarity(self, ref_patches, deg_patches, L):
    # Patches have shape (batch, patch_idx, freq, patch)
    ref_flat = tf.reshape(ref_patches, (-1, _NUM_BANDS, _PATCH_SIZE))
    deg_flat = tf.reshape(deg_patches, (-1, _NUM_BANDS, _PATCH_SIZE))
    nsim_flat = nsim(ref_flat, deg_flat, L)

    batch_nsim = tf.reshape(nsim_flat, (-1, tf.shape(ref_patches)[1]))
    vnsim = tf.reduce_mean(batch_nsim, axis=[1])
    return vnsim


  def align_degraded_patches_audio(self, img_dsig, patches, warp, refPatchIdxs, L):
    raise NotImplementedError
