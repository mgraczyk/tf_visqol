import os
import numpy as np
import soundfile
import subprocess
import json
import gzip
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from atomicwrites import AtomicWriter


def get_tmpdir_on_same_fs(path):
  if path.startswith("/mnt"):
    return "/mnt/tmp"
  elif path.startswith("/datadrive"):
    return "/datadrive/tmp"
  else:
    return "/tmp"


def atomic_write_on_tmp(path, **kwargs):
  writer = AtomicWriter(path, **kwargs)
  return writer._open(lambda: writer.get_fileobject(dir=get_tmpdir_on_same_fs(path)))

def rm_not_exists_ok(path):
  try:
    os.unlink(path)
  except OSError:
    if os.path.exists(path):
      raise


def get_top_level_path(data_path, subdir, input_path):
  rest = input_path.relative_to(data_path)
  return str(Path(data_path, subdir, *rest.parts[1:]))


def get_reference_path(data_path, input_path):
  return get_top_level_path(data_path, "mono_16k_reference",
                            Path(input_path).with_suffix(".wav"))


def get_opus_path(data_path, input_path):
  return get_top_level_path(data_path, "mono_16k_opus_low",
                            Path(input_path).with_suffix(".wav"))

def resample(original, fs_old, fs_new):
  with NamedTemporaryFile(suffix=".wav") as f_in, \
       NamedTemporaryFile(suffix=".wav") as f_out:
    # Scale to avoid clipping.
    original *= (0.5 / (np.max(np.abs(original))))
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

def opus_transcode(input_path, output_path, bitrate=2):
  with atomic_write_on_tmp(output_path, overwrite=True) as output_f, \
       NamedTemporaryFile() as opus_file:
    # Create Opus encoded audio.
    subprocess.check_call([
      "opusenc", "--quiet", "--max-delay", "0", "--padding", "0", "--bitrate",
      str(bitrate), input_path, opus_file.name
    ])

    # Decode back to WAV.
    subprocess.check_call(
      ["opusdec", "--quiet", "--float", "--force-wav", opus_file.name, output_f.name])

def awgn_at_signal_level(x, p):
  x_pow = np.sqrt(np.mean(x**2))
  return p * x_pow * np.random.randn(len(x))

def squishyball(fs, *signals, names=None):
  names = names or [""]*len(signals)

  with ExitStack() as stack:
    temp_files = [
      stack.enter_context(NamedTemporaryFile(suffix="{}{} {}.wav".format("\b"*15, " "*15, name)))
      for name, _ in zip(names, signals)
    ]
    for s, tf in zip(signals, temp_files):
      soundfile.write(tf.name, s, fs, format="wav", subtype="float")
    subprocess.check_call(["squishyball"] + [tf.name for tf in temp_files])

def load_index(index_path):
  with gzip.open(index_path, "rt") as gz_f:
    return json.load(gz_f)
