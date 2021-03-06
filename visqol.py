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

_NUM_BANDS = np.arange(len(_BFS))
_PATCH_SIZE = 30
_PI = np.pi


def interpolate_patch(image_patch):
  # Not used (or properly implemented) because warping is ignored.
  nrows, ncols = img_patch.shape
  src_img_patch = img_patch

  wcols = ncols
  x1 = np.arange(wcols)
  vec = np.arange(nrows)
  img_patch = interp2d(np.arange(ncols), vec, src_img_patch, kind="cubic")(x1, vec)


# Adapted from
#   http://www.mathworks.com/matlabcentral/fileexchange/35103-generalized-goertzel-algorithm/content/goertzel_general_shortened.m
def gga_freq_abs(x, sample_rate, freq):
  lx = len(x)
  pik_term = 2 * _PI * freq / sample_rate
  cos_pik_term = np.cos(pik_term)
  cos_pik_term2 = 2 * np.cos(pik_term)

  # number of iterations is (by one) less than the length of signal
  # Pipeline the first two iterations.
  s1 = x[0]
  s0 = x[1] + cos_pik_term2 * s1
  s2 = s1
  s1 = s0
  for ind in range(2, lx - 1):
    s0 = x[ind] + cos_pik_term2 * s1 - s2
    s2 = s1
    s1 = s0

  s0 = x[lx - 1] + cos_pik_term2 * s1 - s2

  # | s0 - s1 exp(-ip) |
  # | s0 - s1 cos(p) + i s1 sin(p)) |
  # sqrt((s0 - s1 cos(p))^2 + (s1 sin(p))^2)
  y = np.sqrt((s0 - s1*cos_pik_term)**2 + (s1 * np.sin(pik_term))**2)
  # y = np.sqrt(s0**2 + s1**2 - s0*s1*cos_pik_term2)
  return y


def spectrogram_abs(x, window, window_overlap, bfs, fs):
  n = x.shape[0]
  num_blocks = n // window_overlap - 1
  S = np.empty((len(bfs), num_blocks), dtype=np.float64)

  for i in range(num_blocks):
    block = window * x[i * window_overlap: i * window_overlap + len(window)]
    S[:, i] = gga_freq_abs(block, fs, bfs)

  return S

def filter2(h, X, shape):
  assert shape == "valid"

  # The MATLAB version truncates the border.
  result = scipy.ndimage.convolve(X, h, mode="constant", cval=0)
  result = result[1:-1, 1:-1]
  return result

def nsim(neuro_r, neuro_d, L):
  window = np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838],
                     [0.0113, 0.0838, 0.0113]])
  window = window / np.sum(window)

  K1 = 0.01
  K2 = 0.03
  C1 = (K1 * L)**2
  C2 = ((K2 * L)**2) / 2

  neuro_r = neuro_r.astype(np.float64, copy=False)
  neuro_d = neuro_d.astype(np.float64, copy=False)
  mu_r = filter2(window, neuro_r, 'valid')
  mu_d = filter2(window, neuro_d, 'valid')
  mu_r_sq = mu_r * mu_r
  mu_d_sq = mu_d * mu_d
  mu_r_mu_d = mu_r * mu_d
  sigma_r_sq = filter2(window, neuro_r * neuro_r, 'valid') - mu_r_sq
  sigma_d_sq = filter2(window, neuro_d * neuro_d, 'valid') - mu_d_sq
  sigma_r_d = filter2(window, neuro_r * neuro_d, 'valid') - mu_r_mu_d
  sigma_r = np.sign(sigma_r_sq) * np.sqrt(np.abs(sigma_r_sq))
  sigma_d = np.sign(sigma_d_sq) * np.sqrt(np.abs(sigma_d_sq))
  L_r_d = (2 * mu_r * mu_d + C1) / (mu_r_sq + mu_d_sq + C1)
  S_r_d = (sigma_r_d + C2) / (sigma_r * sigma_d + C2)
  nmap = L_r_d * S_r_d

  mNSIM = np.mean(nmap)
  return mNSIM

class Visqol(object):
  def __init__(self, fs):
    self._fs = fs
    if self._fs != 16000:
      raise NotImplementedError

    window_size = round((self._fs / 8000) * 256)
    window_size = 2 * (window_size // 2)

    self._window = np.hamming(window_size + 1)[:window_size]

  def visqol(self, ref, deg):
    img_rsig = self.get_sig_spect(ref)
    img_dsig = self.get_sig_spect(deg)

    lowfloor = np.min(img_rsig)
    img_rsig = img_rsig - lowfloor
    img_dsig = img_dsig - lowfloor
    L = 160

    patches, refPatchIdxs = self.create_ref_patches(img_rsig)

    # TODO: Implement
    # degPatchIdxs = self.align_degraded_patches_audio(img_dsig, patches, refPatchIdxs, L)
    # replace_idx = np.abs(refPatchIdxs - degPatchIdxs) > 30
    # degPatchIdxs[replace_idx] = refPatchIdxs[replace_idx]

    degPatchIdxs = refPatchIdxs

    nsim = self.calc_patch_similarity(patches, degPatchIdxs, img_dsig, L)

    return nsim

  def get_sig_spect(self, x):
    window_overlap = int(len(self._window)*0.5)
    S = spectrogram_abs(x, self._window, window_overlap, _BFS, self._fs)

    S = np.maximum(S, 1e-20)
    S /= np.max(S)
    spec_bf = 20*np.log10(S)
    return spec_bf

  def create_ref_patches(self, img_rsig):
    ref_patch_idxs = range(
      int(_PATCH_SIZE / 2) - 1, img_rsig.shape[1] - _PATCH_SIZE, _PATCH_SIZE)
    num_patches = len(ref_patch_idxs)

    patches = [
      img_rsig[:, ref_patch_idxs[fidx]:ref_patch_idxs[fidx] + _PATCH_SIZE]
      for fidx in range(num_patches)
    ]

    return patches, ref_patch_idxs


  def align_degraded_patches_audio(self, img_dsig, patches, warp, refPatchIdxs, L):
    raise NotImplementedError


  def calc_patch_similarity(self, patches, degPatchIdxs, img_dsig, L):
    NUM_PATCHES = len(patches)
    mwxp = np.zeros(NUM_PATCHES)

    for fidx in range(NUM_PATCHES):
      slide_offset = degPatchIdxs[fidx]
      img_patch = patches[fidx]
      if slide_offset + img_patch.shape[1] < len(img_dsig[0, :]):
        neuro_r = img_patch

        # TODO(mgraczyk): These indices could be wrong
        start = slide_offset
        end = slide_offset + img_patch.shape[1]
        neuro_d = img_dsig[:, slide_offset:end]
        mwxp[fidx] = nsim(neuro_r, neuro_d, L)

    patchNSIM = mwxp
    vnsim = np.mean(patchNSIM)
    return vnsim
