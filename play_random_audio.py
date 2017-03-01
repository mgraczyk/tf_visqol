#!/usr/bin/env python3
import argparse
import sys
import soundfile
import gzip
import threading
import json
import tensorflow as tf
import numpy as np
from itertools import count
from pathlib import Path
from queue import Queue

from tf_visqol import TFVisqol, _DTYPE
from simple_model import get_simple_model, get_loss
from train_simple_model import load_index, load_data_forever
from util import squishyball
from logger import logger

_FS = 16000
_BATCH_SIZE = 1

def load_index(index_path):
  with gzip.open(index_path) as gz_f:
    return json.load(gz_f)

def load_data_forever(index, train_data_queue):
  num_infos = index["count"]
  infos = index["infos"]
  assert num_infos <= len(infos)

  block_size = index["block_size"]
  while True:
    data_indices = np.random.choice(num_infos, size=_BATCH_SIZE, replace=False)
    ref_batch = np.empty((_BATCH_SIZE, block_size), dtype=np.float32)
    deg_batch = np.empty((_BATCH_SIZE, block_size), dtype=np.float32)

    for i, data_idx in enumerate(data_indices):
      info = infos[data_idx]
      start = info["start"]
      length = info["length"]

      with soundfile.SoundFile(info["ref"]) as ref_sf:
        ref_sf.seek(start)
        ref_sf.read(frames=length, dtype=np.float32, out=ref_batch[i])

      with soundfile.SoundFile(info["deg"]) as deg_sf:
        deg_sf.seek(start)
        deg_sf.read(frames=length, dtype=np.float32, out=deg_batch[i])

    train_data_queue.put((ref_batch, deg_batch))


def get_arg_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "index_path", help="The path to the data index created with index_data.py.")
  parser.add_argument("model_checkpoint_path", help="Path to the model checkpoint file")
  parser.add_argument(
    "--no-loss", action="store_true", default=False, help="Do not compute or show losses")

  return parser

def main(argv):
  opts = get_arg_parser().parse_args(argv[1:])
  index = load_index(opts.index_path)
  compute_loss = not opts.no_loss

  logger.info("Starting 1 data thread")
  train_data_queue = Queue(8)
  data_thread = threading.Thread(target=load_data_forever, args=(index, train_data_queue))
  data_thread.start()

  logger.info("Building model")
  block_size = index["block_size"]
  ref_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="ref")
  deg_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="deg")

  filter_output_var = get_simple_model(deg_var, block_size)
  if compute_loss:
    loss_var = get_loss(ref_var, filter_output_var, _FS, block_size)

  with tf.Session() as sess:
    all_vars = tf.global_variables()
    logger.info("Restoring {}".format([v.name for v in all_vars]))
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, opts.model_checkpoint_path)

    for i in count():
      logger.info("Getting data for {}".format(i))
      ref_batch, deg_batch = train_data_queue.get()
      feed_dict = {ref_var: ref_batch, deg_var: deg_batch}

      logger.info("Running batch {}".format(i))
      if compute_loss:
        loss, filter_output = sess.run([loss_var, filter_output_var], feed_dict)
        logger.info("Loss is {}".format(loss))
      else:
        filter_output = sess.run(filter_output_var, feed_dict)

      logger.info("Playing reference, degraded, filter output")
      squishyball(_FS, ref_batch.T, deg_batch.T, filter_output.T)


if __name__ == "__main__":
  main(sys.argv)
