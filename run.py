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

from tf_visqol import TFVisqol


def resample(original, fs_old, fs_new):
  with NamedTemporaryFile(suffix=".wav") as f_in, \
       NamedTemporaryFile(suffix=".wav") as f_out:
    soundfile.write(f_in, original, fs_old)
    f_in.flush()

    # Use R to seed the resample dither rng.
    subprocess.check_call(("sox", "-R", f_in.name, "-r", str(fs_new), f_out.name))
    resampled, new_fs = soundfile.read(f_out.name)
    assert new_fs == fs_new
    return resampled

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

  tf_visqol = TFVisqol(fs)
  test = tf_visqol.visqol(x, with_awgn)

  return [gold, test]

def main(argv):
  original, fs_old = soundfile.read("original.wav")

  fs = 16000
  original = resample(original, fs_old, fs)

  noise_powers = np.logspace(-3, 0, 10);

  x = np.squeeze(original[:, 0])
  func = functools.partial(visqol_with_awgn, x, fs)
  with Pool(processes=4) as pool:
    nsim = np.array(pool.map(func, noise_powers))
  # nsim = np.array(list(map(func, noise_powers)))

  print(nsim)
  # plt.plot(noise_powers, nsim)
  # plt.show()


if __name__ == "__main__":
  main(sys.argv)
