import os
import numpy as np
import soundfile
import subprocess
from tempfile import NamedTemporaryFile
from atomicwrites import AtomicWriter


def get_tmpdir_on_same_fs(path):
  if path.startswith("/mnt"):
    return "/mnt/tmp"
  else:
    return "/tmp"


def atomic_write_on_tmp(path, **kwargs):
  writer = AtomicWriter(path, **kwargs)
  return writer._open(lambda: writer.get_fileobject(dir=get_tmpdir_on_same_fs(path)))


def resample(original, fs_old, fs_new):
  with NamedTemporaryFile(suffix=".wav") as f_in, \
       NamedTemporaryFile(suffix=".wav") as f_out:
    soundfile.write(f_in, original, fs_old)
    f_in.flush()

    # Use R to seed the resample dither rng.
    # Scale with -v to avoid clipping.
    subprocess.check_call(("sox", "-v", str(63/64), "-R", f_in.name, "-r", str(fs_new), f_out.name))
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

def opus_transcode(input_path, output_path, bitrate=2):
  with atomic_write_on_tmp(output_path, overwrite=True) as output_f, \
       NamedTemporaryFile() as opus_file:
    # Create Opus encoded audio.
    subprocess.check_call(
      ["opusenc", "--quiet", "--bitrate", str(bitrate), input_path, opus_file.name])

    # Decode back to WAV.
    subprocess.check_call(
      ["opusdec", "--quiet", "--force-wav", opus_file.name, output_f.name])

def awgn_at_signal_level(x, p):
  x_pow = np.sqrt(np.mean(x**2))
  return p * x_pow * np.random.randn(len(x))
