import numpy as np
import soundfile
from tempfile import NamedTemporaryFile
import subprocess


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

def visqol_matlab(ref, deg, fs):
  with NamedTemporaryFile(suffix=".wav") as f_ref, \
       NamedTemporaryFile(suffix=".wav") as f_deg:
    soundfile.write(f_ref, ref, fs)
    f_ref.flush()
    soundfile.write(f_deg, deg, fs)
    f_deg.flush()
    args = ("./visqol", f_ref.name, f_deg.name)
    output = subprocess.check_output(args, cwd=".")
    return float(output)

def awgn_at_signal_level(x, p):
  x_pow = np.sqrt(np.mean(x**2))
  return p * x_pow * np.random.randn(len(x))
