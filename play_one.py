#!/usr/bin/env python3
import argparse
import sys
import soundfile
from tempfile import NamedTemporaryFile
from queue import Queue

from train_simple_model import load_index
from play_random_audio import run_play_audio
from util import resample, opus_transcode
from logger import logger

_FS = 16000

def get_arg_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "input_path", help="The path to the data index created with index_data.py.")

  parser.add_argument("model_checkpoint_path", help="Path to the model checkpoint file")
  parser.add_argument(
    "--no-loss", action="store_true", default=False, help="Do not compute or show losses")

  return parser


def main(argv):
  opts = get_arg_parser().parse_args(argv[1:])
  train_data_queue = Queue(1)

  original, original_fs = soundfile.read(opts.input_path)
  original_mono = original[:, 0] if len(original.shape) > 1 else original

  ref = resample(original_mono, original_fs, _FS)
  with NamedTemporaryFile(suffix=".wav") as resampled_f, \
       NamedTemporaryFile(suffix=".wav") as opus_f:
    soundfile.write(resampled_f.name, ref, samplerate=_FS, format="wav", subtype="float")
    opus_transcode(resampled_f.name, opus_f.name)
    deg, _ = soundfile.read(opus_f.name)

  block_size = 16000
  train_data_queue.put((ref[:block_size, None].T, deg[:block_size, None].T))
  run_play_audio(train_data_queue, block_size, opts)


if __name__ == "__main__":
  main(sys.argv)