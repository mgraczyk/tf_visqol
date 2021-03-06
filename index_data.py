#!/usr/bin/env python3
import sys
import argparse
import soundfile
import concurrent
import json
import gzip
import traceback
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from util import atomic_write_on_tmp, find_recursive
from util import get_reference_path, get_opus_path
from logger import logger

_FS = 16000

def _get_info_for_path(data_path, input_path, block_size, overlap):
  reference_path = get_reference_path(data_path, input_path)
  opus_path = get_opus_path(data_path, input_path)
  if not (Path(reference_path).exists() and Path(opus_path).exists()):
    raise Exception("Path {} and {} do not both exist".format(reference_path, opus_path))

  reference_info = soundfile.info(reference_path)
  opus_info = soundfile.info(opus_path)

  if reference_info.format not in ("WAV", "WAVEX"):
    raise Exception(reference_info.format)
  if opus_info.format not in ("WAV", "WAVEX"):
    raise Exception(opus_info.format)

  if reference_info.subtype != "FLOAT":
    raise Exception(reference_info.subtype)
  if opus_info.subtype != "FLOAT":
    raise Exception(opus_info.subtype)

  if reference_info.samplerate != _FS:
    raise Exception(reference_info.samplerate)
  if opus_info.samplerate != _FS:
    raise Exception(opus_info.samplerate)

  num_frames = reference_info.frames
  if opus_info.frames != num_frames:
    raise Exception(opus_info.frames)

  # Only include whole blocks.
  return [{
    "ref": str(Path(reference_path).relative_to(data_path)),
    "deg": str(Path(opus_path).relative_to(data_path)),
    "start": start,
    "length": block_size,
  } for start in range(0, num_frames - block_size + 1, overlap)]


def _get_info_for_paths(data_path, input_paths, block_size, overlap):
  successes = 0
  infos = []
  with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_args = {
      executor.submit(_get_info_for_path, data_path, p, block_size, overlap): p
      for p in input_paths
    }
    for future in tqdm(concurrent.futures.as_completed(future_to_args), total=len(input_paths)):
      input_path = future_to_args[future]
      try:
        info_for_path = future.result()
        infos.extend(info_for_path)
        successes += 1
      except Exception as exc:
        logger.error("{} generated an exception: ".format(input_path))
        traceback.print_exc()
      else:
        logger.debug("Done with {}".format(input_path))
    return infos, successes

def get_arg_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    "data_path", nargs="?",
    default="/Users/mgraczyk/dev/tf_visqol/data",
    help="The root path where the S3 mirror data tree is located.")

  parser.add_argument(
    "-n",
    "--index_name",
    default=str(uuid4()),
    help=("The name of the index."
          " The index will be written to a file data_path/<index_name>.json"))

  parser.add_argument(
    "--block_size",
    type=int,
    default=16000,
    help="The number of samples in each indexed block.")

  parser.add_argument(
    "--overlap",
    type=int,
    default=8000,
    help="The number of samples of overlap between adjacent indexed blocks.")

  return parser


def main(argv):
  opts = get_arg_parser().parse_args(argv[1:])

  input_path_prefix = "{}/original/".format(opts.data_path)
  wav_paths = find_recursive(input_path_prefix, "*.wav")
  flac_paths = find_recursive(input_path_prefix, "*.flac")
  original_paths = wav_paths + flac_paths
  info_for_paths, successes = _get_info_for_paths(opts.data_path, original_paths,
                                                  opts.block_size, opts.overlap)
  index = {
    "count": len(info_for_paths),
    "block_size": opts.block_size,
    "overlap": opts.overlap,
    "infos": info_for_paths
  }

  index_path = str(Path(opts.data_path, opts.index_name + ".json.gz"))
  logger.info("Writing index to {}".format(index_path))
  with atomic_write_on_tmp(index_path, overwrite=True) as f:
    with gzip.open(f.name, "wt") as gz_f:
      json.dump(index, gz_f)

  logger.info("Processed {} examples from {} files with {} inputs".format(
    len(index["infos"]), successes, len(original_paths)))

if __name__ == "__main__":
  main(sys.argv)
