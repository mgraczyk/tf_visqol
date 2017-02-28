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
