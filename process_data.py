#!/usr/bin/env python3
"""Creates resampled and transcoded copies of all data in a directory."""
import soundfile
import concurrent
import glob
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from util import resample
from util import atomic_write_on_tmp
from util import opus_transcode
from logger import logger

_FS = 16000


def get_top_level_path(data_path, subdir, input_path):
  rest = Path(input_path).relative_to(data_path)
  return str(Path(data_path, subdir, *rest.parts[1:]))


def get_reference_path(data_path, input_path):
  return get_top_level_path(data_path, "mono_16k_reference", input_path)


def get_opus_path(data_path, input_path):
  return get_top_level_path(data_path, "mono_16k_opus_low", input_path)


def process_input(input_path, reference_path, opus_path):
  if Path(reference_path).exists() and Path(opus_path).exists():
    logger.info("Paths {} and {} exist, skipping".format(reference_path, opus_path))
    return
  else:
    logger.info("Paths {} and {} don't exist, processing".format(reference_path, opus_path))
  assert Path(input_path).exists()


  original, original_fs = soundfile.read(input_path)
  original_mono = original[:, 0] if len(original.shape) > 1 else original

  Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
  resampled = resample(original_mono, original_fs, _FS)
  with atomic_write_on_tmp(reference_path, overwrite=True) as f:
    soundfile.write(f.name, resampled, samplerate=_FS, format="wav", subtype="float")

  Path(opus_path).parent.mkdir(parents=True, exist_ok=True)
  opus_transcode(reference_path, opus_path)


def run_on_paths(data_path, input_paths):
  work_args = ((input_path, get_reference_path(data_path, input_path),
                get_opus_path(data_path, input_path)) for input_path in input_paths)
  successes = 0
  with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_args = {executor.submit(process_input, *args): args for args in work_args}
    for future in tqdm(concurrent.futures.as_completed(future_to_args)):
      input_path, reference_path, opus_path = future_to_args[future]
      try:
        data = future.result()
        successes += 1
      except Exception as exc:
        raise
        logger.error("{} generated an exception: ".format(input_path))
        traceback.print_exc()
      else:
        logger.info("Done with {}".format(input_path))
    return successes


def main():
  data_path = "/Users/mgraczyk/dev/tf_visqol/data"
  input_paths = glob.glob("{}/original/**/*.wav".format(data_path), recursive=True)

  num_run = run_on_paths(data_path, input_paths)
  logger.info("Done processing {} with {} inputs".format(num_run, len(input_paths)))

if __name__ == "__main__":
  main()
