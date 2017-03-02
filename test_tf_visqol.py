#!/usr/bin/env python3
import sys
import numpy as np
import soundfile
import functools
import random
from multiprocessing import Pool

import matplotlib
from matplotlib import pyplot as plt

from tf_visqol import TFVisqol
from util import resample
from util import awgn_at_signal_level
from util import visqol_matlab

# HACK to get around not being able to pickle a lambda or partial out of order.
def visqol_matlab_reordered(ref, fs, deg):
  return visqol_matlab(ref, deg, fs)

def main(argv):
  original, fs_old = soundfile.read("original.wav")

  fs = 16000
  original = resample(original, fs_old, fs)
  original = original[:, 0]

  noise_powers = np.logspace(-3, 0, 10)
  np.random.seed(1)
  with_awgn = np.stack([original + awgn_at_signal_level(original, p) for p in noise_powers])

  print("Running Tensorflow")
  test = TFVisqol(fs).visqol_with_session( np.broadcast_to(original, (with_awgn.shape[0], original.shape[0])), with_awgn)

  use_precomputed_gold = True
  if use_precomputed_gold:
    print("Using precomputed golden value")
    gold = [
      0.996206, 0.987429, 0.970375, 0.932609, 0.875888, 0.810587, 0.737579, 0.666732,
      0.582871, 0.521061
    ]
  else:
    print("Running MATLAB")
    func = functools.partial(visqol_matlab_reordered, original, fs)
    with Pool(processes=4) as pool:
      gold = np.array(pool.map(func, with_awgn))

  diff = np.abs(gold - test)

  print(np.column_stack((gold, test)))
  assert np.all(diff < 1e-3), "Tensorflow implementation does not match MATLAB"
  # plt.plot(noise_powers, nsim)
  # plt.show()
  print("PASS\n")


if __name__ == "__main__":
  main(sys.argv)
