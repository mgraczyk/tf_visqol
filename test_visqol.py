#!/usr/bin/env python3
import sys
import numpy as np
import soundfile
import subprocess
import functools
import random
from tempfile import NamedTemporaryFile
from multiprocessing import Pool

import matplotlib
from matplotlib import pyplot as plt

from visqol import Visqol
from tf_visqol import TFVisqol
from util import resample

def visqol_simple(ref, deg, fs):
  with NamedTemporaryFile(suffix=".wav") as f_ref, \
       NamedTemporaryFile(suffix=".wav") as f_deg:
    soundfile.write(f_ref, ref, fs)
    f_ref.flush()
    soundfile.write(f_deg, deg, fs)
    f_deg.flush()
    args = ("./visqol", f_ref.name, f_deg.name)
    output = subprocess.check_output(args, cwd=".")
    return float(output)

def visqol_with_awgn(x, fs, p):
  np.random.seed(1)
  x_pow = np.sqrt(np.mean(x**2))
  with_awgn = x + p * x_pow * np.random.randn(len(x))
  gold = visqol_simple(x, with_awgn, fs)

  test = Visqol(fs).visqol(x, with_awgn)
  test_tf = TFVisqol(fs).visqol_with_session(x, with_awgn)

  return [gold, test, test_tf]

def main(argv):
  original, fs_old = soundfile.read("original.wav")

  fs = 16000
  original = resample(original, fs_old, fs)
  original = original[16000:32000, :]

  noise_powers = np.logspace(-3, 0, 10);
  # noise_powers = np.logspace(-3, 0, 1);

  x = np.squeeze(original[:, 0])
  func = functools.partial(visqol_with_awgn, x, fs)
  with Pool(processes=4) as pool:
    nsim = np.array(pool.map(func, noise_powers))
  # nsim = np.array(list(map(func, noise_powers)))

  diff = np.abs(nsim[:, 0] - nsim[:, 1])
  diff_tf = np.abs(nsim[:, 0] - nsim[:, 2])

  print(nsim)
  assert np.all(diff < 1e-3), "Python implementation does not match MATLAB"
  assert np.all(diff_tf < 1e-3), "Tensorflow implementation does not match MATLAB"
  # plt.plot(noise_powers, nsim)
  # plt.show()
  print("PASS\n")


if __name__ == "__main__":
  main(sys.argv)
