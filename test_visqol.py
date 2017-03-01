#!/usr/bin/env python3
import sys
import numpy as np
import soundfile
import functools
import random
from multiprocessing import Pool

import matplotlib
from matplotlib import pyplot as plt

from visqol import Visqol
from util import resample, awgn_at_signal_level, visqol_matlab

def visqol_with_awgn(x, fs, p):
  np.random.seed(1)
  with_awgn = x + awgn_at_signal_level(x, p)

  gold = visqol_matlab(x, with_awgn, fs)
  test = Visqol(fs).visqol(x, with_awgn)

  return [gold, test]

def main(argv):
  original, fs_old = soundfile.read("original.wav")

  fs = 16000
  original = resample(original, fs_old, fs)

  noise_powers = np.logspace(-3, 0, 10);
  x = original[:2*fs, 0]
  func = functools.partial(visqol_with_awgn, x, fs)
  with Pool(processes=4) as pool:
    nsim = np.array(pool.map(func, noise_powers))
  # nsim = np.array(list(map(func, noise_powers)))

  diff = np.abs(np.diff(nsim, axis=1))

  print(nsim)
  assert np.all(diff < 1e-3), "Python implementation does not match MATLAB"
  # plt.plot(noise_powers, nsim)
  # plt.show()
  print("PASS\n")


if __name__ == "__main__":
  main(sys.argv)
