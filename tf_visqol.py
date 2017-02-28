import math
import numpy as np
import tensorflow as tf
import scipy
import scipy.ndimage
from scipy.interpolate import interp2d


_BFS = np.asarray(
  [
    50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900,
    3400, 4000, 4800, 6500, 8000
  ],
  dtype=np.float64)

_NUM_BANDS = len(_BFS)
_PATCH_SIZE = 30
_PI = np.pi
_BLOCK_SIZE = 512

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

# Adapted from
#   http://www.mathworks.com/matlabcentral/fileexchange/35103-generalized-goertzel-algorithm/content/goertzel_general_shortened.m
def gga_freq_abs(x, sample_rate, freq):
  """Computes the magnitude of the time domain signal x at each frequency in freq using
     the generalized Goertzel algorithm.

     x has shape (batch, block)
  """
  lx = _BLOCK_SIZE
  pik_term = 2 * _PI * freq / sample_rate
  cos_pik_term = np.cos(pik_term)
  cos_pik_term2 = 2 * np.cos(pik_term)

  batch_size = 1
  s0 = np.zeros((batch_size, len(freq)), dtype=np.float64)
  s1 = np.zeros((batch_size, len(freq)), dtype=np.float64)
  s2 = np.zeros((batch_size, len(freq)), dtype=np.float64)

  # number of iterations is (by one) less than the length of signal
  for ind in range(lx - 1):
    s0 = x[:, ind] + cos_pik_term2 * s1 - s2
    s2 = s1
    s1 = s0

  s0 = x[:, lx - 1] + cos_pik_term2 * s1 - s2

  # | s0 - s1 exp(-ip) |
  # | s0 - s1 cos(p) + i s1 sin(p)) |
  # sqrt((s0 - s1 cos(p))^2 + (s1 sin(p))^2)
  y = tf.sqrt((s0 - s1*cos_pik_term)**2 + (s1 * np.sin(pik_term))**2)
  return y


def spectrogram_abs(x, window, window_overlap, bfs, fs):
  batch_size = 1

  # TODO: We may need to pad for the last block.
  # TODO: Fix for batches of more than 1.
  x_as_image = tf.expand_dims(tf.expand_dims(x, 1), -1)
  blocks_raw = tf.extract_image_patches(
    x_as_image,
    ksizes=[1, 1, _BLOCK_SIZE, 1],
    strides=[1, 1, window_overlap, 1],
    rates=[1, 1, 1, 1],
    padding="VALID")
  blocks = tf.squeeze(blocks_raw, [1])

  windows_blocks = window * blocks

  # map_fn works along dimension 0.
  perm = [1, 0, 2]
  wb_with_blocks_first = tf.transpose(windows_blocks, perm=perm)
  func = lambda x: gga_freq_abs(x, fs, bfs)
  S_with_blocks_first = tf.map_fn(func, wb_with_blocks_first)
  S = tf.transpose(
    tf.transpose(S_with_blocks_first, perm=tf.invert_permutation(perm)), perm=[0, 2, 1])

  return S

def filter2(h, X, shape):
  # The MATLAB version truncates the border.
  shape = shape.upper()
  assert shape == "VALID"

  # The original performs correlation, this is convolution.
  # The difference doesn't matter because the filter is rotationally symmetric.

  X = tf.expand_dims(X, -1)

  # TODO Tensorflow doesn't support 64-bit conv.
  result = tf.nn.conv2d(tf.to_float(X), tf.to_float(h), strides=[1, 1, 1, 1], padding=shape)
  result = tf.squeeze(result, [-1])
  result = tf.to_double(result)
  return result

def nsim(neuro_r, neuro_d, L):
  # neuro_r and neuro_d are Tensors of (batch, freq, patch)
  window = np.array([[0.0113, 0.0838, 0.0113],
                     [0.0838, 0.6193, 0.0838],
                     [0.0113, 0.0838, 0.0113]]).reshape(3, 3, 1, 1)
  window = window / np.sum(window)
  window = tf.constant(window)

  K1 = 0.01
  K2 = 0.03
  C1 = (K1 * L)**2
  C2 = ((K2 * L)**2) / 2

  # Use double precision.
  neuro_r = tf.to_double(neuro_r)
  neuro_d = tf.to_double(neuro_d)

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
    assert window_size == round((self._fs / 8000) * 256)
    window_size = 2 * (window_size // 2)

    self._window_overlap = window_size / 2
    self._window = tf.constant(np.hamming(window_size + 1)[:window_size])

  def visqol(self, ref, deg):
    with tf.Session() as sess, \
         tf.variable_scope("visqol"):
      ref_var, deg_var, nsim_var = self._visqol_op(len(ref))

      feed_dict = {ref_var: ref.reshape(1, -1), deg_var: deg.reshape(1, -1)}
      nsim = sess.run(nsim_var, feed_dict)
      return nsim[0]

  def _visqol_op(self, n):
    ref = tf.placeholder(tf.float64, (None, n,), name="ref")
    deg = tf.placeholder(tf.float64, (None, n,), name="deg")

    # Images have shape (batch,
    img_rsig = tf.identity(self._get_sig_spect(ref), name="img_rsig")
    img_dsig = tf.identity(self._get_sig_spect(deg), name="img_dsig")

    lowfloor = tf.reduce_min(img_rsig)
    img_rsig = img_rsig - lowfloor
    img_dsig = img_dsig - lowfloor
    L = 160

    ref_patches = tf.identity(self.create_patches(img_rsig), name="ref_patches")
    deg_patches = tf.identity(self.create_patches(img_dsig), name="deg_patches")
    nsim = self.calc_patch_similarity(ref_patches, deg_patches, L)

    return ref, deg, nsim

  def _get_sig_spect(self, x):
    S = spectrogram_abs(x, self._window, self._window_overlap, _BFS, self._fs)

    S = tf.maximum(S, tf.constant(1e-20, dtype=np.float64))
    max_S = tf.reduce_max(S)
    S /= max_S
    spec_bf = 20*log10(S)
    return spec_bf

  def create_patches(self, img_rsig):
    # TODO: This slice is done in the MATLAB, but seems dumb.
    begin = int(_PATCH_SIZE / 2) - 1
    img_rsig = tf.slice(img_rsig, begin=[0, 0, begin], size=[-1, -1, -1])
    img_4d = tf.expand_dims(img_rsig, -1)
    patches = tf.extract_image_patches(
      img_4d,
      ksizes=[1, 1, _PATCH_SIZE, 1],
      strides=[1, 1, _PATCH_SIZE, 1],
      rates=[1, 1, 1, 1],
      padding="VALID")
    patches = tf.transpose(patches, perm=[0, 2, 1, 3])

    return patches


  def calc_patch_similarity(self, ref_patches, deg_patches, L):
    # Patches have shape (batch, patch_idx, freq, patch)
    # map_fn works along dimension 0, so put patch dimension there.
    perm = [1, 0, 2, 3]
    neuro_r = tf.transpose(ref_patches, perm=perm)
    neuro_d = tf.transpose(deg_patches, perm=perm)

    # TODO: We may be able to do this without map_fn by using conv channels.
    func = lambda x: nsim(*tf.unstack(x, num=2, axis=0), L)
    patch_nsim = tf.map_fn(func, tf.stack((neuro_r, neuro_d), axis=1))
    batch_nsim = tf.transpose(patch_nsim, [1, 0])

    # TODO(mgraczyk): Preserve batch axis.
    vnsim = tf.reduce_mean(batch_nsim, axis=[1])
    return vnsim


  def align_degraded_patches_audio(self, img_dsig, patches, warp, refPatchIdxs, L):
    raise NotImplementedError
