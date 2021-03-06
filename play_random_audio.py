#!/usr/bin/env python3
import argparse
import sys
import threading
import tensorflow as tf
import numpy as np
from pathlib import Path
from itertools import count
from queue import Queue

import matplotlib
from matplotlib import pyplot as plt

from tf_visqol import _DTYPE
from visqol import Visqol
from simple_model import get_simple_model
from train_simple_model import load_data_forever
from util import squishyball
from util import load_index
from audio_util import windowed_overlap_add
from script_util import get_data_script_arg_parser
from logger import logger

_FS = 16000

def plot_spectrograms(ref, deg, filtered):
  NFFT = 1024
  overlap = 900
  plt.close("all")
  ax1 = plt.subplot(511)
  plt.specgram(ref, NFFT=NFFT, Fs=_FS, noverlap=overlap)
  plt.ylabel("ref")
  plt.subplot(512, sharex=ax1)
  plt.specgram(deg, NFFT=NFFT, Fs=_FS, noverlap=overlap)
  plt.ylabel("deg")
  plt.subplot(513, sharex=ax1)
  plt.specgram(filtered, NFFT=NFFT, Fs=_FS, noverlap=overlap)
  plt.ylabel("filt")
  plt.subplot(514, sharex=ax1)
  plt.specgram(filtered - ref, NFFT=NFFT, Fs=_FS, noverlap=overlap)
  plt.ylabel("filt v. ref")
  plt.subplot(515, sharex=ax1)
  plt.specgram(filtered - deg, NFFT=NFFT, Fs=_FS, noverlap=overlap)
  plt.ylabel("filt v. deg")

  plt.figure()
  ax2 = plt.subplot(411)
  plt.plot(ref)
  plt.ylabel("ref")
  plt.subplot(412, sharex=ax2)
  plt.plot(filtered)
  plt.ylabel("filt")
  plt.subplot(413, sharex=ax2)
  plt.plot(deg)
  plt.ylabel("deg")
  plt.subplot(414, sharex=ax2)
  plt.plot(ref - filtered)
  plt.ylabel("ref - filt")

  plt.show(block=False)
  plt.pause(0.01)


def get_arg_parser():
  parser = get_data_script_arg_parser()
  parser.add_argument("model_checkpoint_path", help="Path to the model checkpoint file")
  parser.add_argument(
    "--no-loss", action="store_true", default=False, help="Do not compute or show losses")
  parser.add_argument(
    "--no-plots", action="store_true", default=False, help="Do not show spectrogram or other plots.")

  return parser

def run_play_audio(train_data_queue, block_size, opts):
  compute_loss = not opts.no_loss
  show_plots = not opts.no_plots

  logger.info("Building model")
  ref_var = tf.placeholder(_DTYPE, (1, block_size), name="ref")
  deg_var = tf.placeholder(_DTYPE, (1, block_size), name="deg")

  filter_output_var = get_simple_model(deg_var, block_size)

  visqol = Visqol(_FS)

  with tf.Session() as sess:
    all_vars = tf.global_variables()
    logger.info("Restoring {}".format([v.name for v in all_vars]))
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, opts.model_checkpoint_path)

    for i in count():
      logger.info("Getting data for {}".format(i))
      ref_batch, deg_batch = train_data_queue.get()
      ref = ref_batch[:1, :]
      deg = deg_batch[:1, :]
      assert ref.shape == deg.shape

      if ref_batch.shape[1] > block_size:
        block_num = [0]
        def _process_block(deg_block):
          logger.info("Running batch {} block {}".format(i, block_num[0]))
          block_num[0] += 1
          return sess.run(filter_output_var, {deg_var: deg_block})
        filter_output = windowed_overlap_add(deg,
                                             np.hamming(block_size + 1)[:block_size],
                                             block_size // 2, _process_block)
      else:
        logger.info("Running batch {}".format(i))
        filter_output = sess.run(filter_output_var, {deg_var: deg})

      ref_flat = ref.reshape(-1)
      deg_flat = deg.reshape(-1)
      filtered_flat = filter_output.reshape(-1)
      if compute_loss:
        original_loss = visqol.visqol(ref_flat, deg_flat)
        filtered_loss = visqol.visqol(ref_flat, filtered_flat)
        logger.info("nsim from {} - {} = {}".format(original_loss, filtered_loss,
                                                    original_loss - filtered_loss))

      logger.info("Playing reference, degraded, filter output")
      logger.info("Mean square difference after filtering is {}".format(
        np.mean(np.square(deg - filter_output))))
      logger.info("Num NaN {}".format(np.sum(np.isnan(filter_output))))
      logger.info("Num clipped ({}, {})".format(
        np.sum(filter_output == -1), np.sum(filter_output == 1)))

      if show_plots:
        plot_spectrograms(ref_flat, deg_flat, filtered_flat)
      squishyball(
        _FS,
        ref.T,
        deg.T,
        filter_output.T,
        ref.T - filter_output.T,
        names=("ref", "deg", "filt", "ref - filt"))

def main(argv):
  opts = get_arg_parser().parse_args(argv[1:])
  index = load_index(opts.index_path)
  data_path = opts.data_path or str(Path(opts.index_path).parent)

  logger.info("Starting 1 data thread")
  train_data_queue = Queue(8)
  data_thread = threading.Thread(target=load_data_forever, args=(data_path, index, train_data_queue))
  data_thread.start()

  run_play_audio(train_data_queue, index["block_size"], opts)


if __name__ == "__main__":
  main(sys.argv)
