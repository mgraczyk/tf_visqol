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
from uuid import uuid4
from pathlib import Path
from queue import Queue

from tf_visqol import _DTYPE
from simple_model import get_simple_model, get_loss, get_minimize_op
from util import rm_not_exists_ok
from logger import logger

_RANDOM_SEED = 42
tf.set_random_seed(_RANDOM_SEED)

_BATCH_SIZE = 16
_FS = 16000

def load_index(index_path):
  with gzip.open(index_path, "rt") as gz_f:
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
  training_id = uuid4()
  logger.info("Training with id {}".format(training_id))

  training_path = Path("./models/{}".format(training_id))
  training_path.mkdir(parents=True, exist_ok=True)
  latest_path = Path("./models/latest")
  rm_not_exists_ok(str(latest_path))
  latest_path.symlink_to(training_path.relative_to(latest_path.parent), True)

  opts = get_arg_parser().parse_args(argv[1:])
  index = load_index(opts.index_path)

  logger.info("Starting 1 data thread")
  train_data_queue = Queue(32)
  data_thread = threading.Thread(target=load_data_forever, args=(index, train_data_queue))
  data_thread.start()

  logger.info("Building model")
  block_size = index["block_size"]
  ref_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="ref")
  deg_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="deg")

  filter_output_var = get_simple_model(deg_var, block_size)
  loss_var = get_loss(ref_var, filter_output_var, _FS, block_size)
  minimize_op = get_minimize_op(loss_var)
  init_op_new = tf.global_variables_initializer()
  init_op_old = tf.initialize_all_variables()

  saver = tf.train.Saver(tf.trainable_variables())

  with tf.Session() as sess:
    # Initialize init_op_new.
    logger.info("Initializing")
    sess.run(init_op_old)
    sess.run(init_op_new)

    for i in count():
      logger.info("Getting data for {}".format(i))
      ref_batch, deg_batch = train_data_queue.get()
      feed_dict = {ref_var: ref_batch, deg_var: deg_batch}

      logger.info("Running batch {}".format(i))
      _, loss = sess.run([minimize_op, loss_var], feed_dict)
      logger.info("Loss is {}".format(loss))

      if i > 0 and i % 100 == 0:
        checkpoint_path = Path(training_path, "model_checkpoint" "{}.ckpt".format(i))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = saver.save(sess, str(checkpoint_path))
        logger.info("Saved model to {}".format(save_path))


if __name__ == "__main__":
  main(sys.argv)
