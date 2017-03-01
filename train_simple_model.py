#!/usr/bin/env python3
import argparse
import sys
import soundfile
import gzip
import threading
import json
import tensorflow as tf
import numpy as np
from queue import Queue
from tqdm import tqdm

from tf_visqol import TFVisqol, _DTYPE
from simple_model import get_simple_model, get_loss
from util import resample
from logger import logger

_RANDOM_SEED = 42
tf.set_random_seed(_RANDOM_SEED)

_BATCH_SIZE = 64
_FS = 16000

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

  return parser

def main(argv):
  opts = get_arg_parser().parse_args(argv[1:])
  index = load_index(opts.index_path)
  block_size = index["block_size"]

  logger.info("Starting 1 data thread")
  train_data_queue = Queue(32)
  data_thread = threading.Thread(target=load_data_forever, args=(index, train_data_queue))
  data_thread.start()

  logger.info("Building model")
  ref_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="ref")
  deg_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="deg")

  filter_output = get_simple_model(deg_var, block_size)
  loss_var, minimize_op = get_loss(ref_var, filter_output, _FS, block_size)
  init_op_new = tf.global_variables_initializer()
  init_op_old = tf.initialize_all_variables()

  with tf.Session() as sess:
    # Initialize init_op_new.
    logger.info("Initializing")
    sess.run(init_op_old)
    sess.run(init_op_new)

    for i in range(1000):
      logger.info("Running batch {}".format(i))
      ref_batch, deg_batch = train_data_queue.get()
      feed_dict = {ref_var: ref_batch, deg_var: deg_batch}
      _, loss = sess.run([minimize_op, loss_var], feed_dict)
      logger.info("Loss is {}".format(loss))

if __name__ == "__main__":
  main(sys.argv)
