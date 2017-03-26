#!/usr/bin/env python3
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
from simple_model import get_baseline_model
from util import rm_not_exists_ok, mkdirs_exists_ok
from util import load_index
from script_util import get_data_script_arg_parser

import logging
from logger import logger

_RANDOM_SEED = 42
tf.set_random_seed(_RANDOM_SEED)

_BATCH_SIZE = 32
_FS = 16000

def load_data_forever(data_path, index, indices, train_data_queue):
  block_size = index["block_size"]
  infos = index["infos"]

  start = indices.size
  while True:
    if start + _BATCH_SIZE > indices.size:
      logger.info("\nShuffing dataset with {} items".format(indices.size))
      np.random.shuffle(indices)
      start = 0

    ref_batch = np.empty((_BATCH_SIZE, block_size), dtype=np.float32)
    deg_batch = np.empty((_BATCH_SIZE, block_size), dtype=np.float32)

    this_batch_indices = indices[start:start + _BATCH_SIZE]
    for i, data_idx in enumerate(this_batch_indices):
      info = infos[data_idx]
      rec_start = info["start"]
      length = info["length"]

      with soundfile.SoundFile(str(Path(data_path, info["ref"]))) as ref_sf:
        ref_sf.seek(rec_start)
        ref_sf.read(frames=length, dtype=np.float32, out=ref_batch[i])

      with soundfile.SoundFile(str(Path(data_path, info["deg"]))) as deg_sf:
        deg_sf.seek(rec_start)
        deg_sf.read(frames=length, dtype=np.float32, out=deg_batch[i])

    train_data_queue.put((ref_batch, deg_batch))
    start += _BATCH_SIZE

def get_loss_logger(training_path):
  loss_logger = logger.getChild("loss")
  loss_logger.handlers = []
  loss_handler = logging.FileHandler(str(Path(training_path, "losses.log")))
  loss_handler.setFormatter(logging.Formatter(""))
  loss_logger.addHandler(loss_handler)
  return loss_logger

def main(argv):
  training_id = uuid4()
  logger.info("Training with id {}".format(training_id))

  training_path = Path("./models/{}".format(training_id))
  mkdirs_exists_ok(str(training_path))
  latest_path = Path("./models/latest")
  rm_not_exists_ok(str(latest_path))
  latest_path.symlink_to(training_path.relative_to(latest_path.parent), True)

  loss_logger = get_loss_logger(training_path)

  opts = get_data_script_arg_parser().parse_args(argv[1:])
  index = load_index(opts.index_path)
  data_path = opts.data_path or str(Path(opts.index_path).parent)

  train_data_queue = Queue(128)
  num_threads = 3

  logger.info("Starting {} data thread(s)".format(num_threads))
  indices = np.arange(len(index["infos"]))
  indices_per_thread = indices.size // num_threads
  for i in range(num_threads - 1):
    threading.Thread(
      target=load_data_forever,
      args=(data_path,
            index, indices[i * indices_per_thread:(i + 1) * indices_per_thread],
            train_data_queue)).start()
  threading.Thread(
    target=load_data_forever,
    args=(data_path, index, indices[(num_threads - 1) * indices_per_thread:],
          train_data_queue)).start()

  logger.info("Building model")
  block_size = index["block_size"]
  ref_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="ref")
  deg_var = tf.placeholder(_DTYPE, (_BATCH_SIZE, block_size), name="deg")

  filter_output_var = get_baseline_model(deg_var, block_size)
  losses = get_loss(ref_var, deg_var, filter_output_var, _FS, block_size)
  minimize_op, opt = get_minimize_op(losses["loss"])
  grads = opt.compute_gradients(losses["nsim"])
  norm_op = tf.global_norm(grads)

  init_op_new = tf.global_variables_initializer()
  init_op_old = tf.initialize_all_variables()

  saver = tf.train.Saver(tf.trainable_variables())

  with tf.Session() as sess, \
       tf.device("/gpu:0"):
    # Initialize init_op_new.
    logger.info("Initializing")
    sess.run(init_op_old)
    sess.run(init_op_new)

    for i in count():
      logger.info("Getting data for {}".format(i))
      ref_batch, deg_batch = train_data_queue.get()
      feed_dict = {ref_var: ref_batch, deg_var: deg_batch}

      logger.info("Running batch {}".format(i))
      _, norm, *loss_values = sess.run([minimize_op, norm_op] + list(losses.values()),
                                       feed_dict)
      loss_logger.info("Losses: {}".format(dict(zip(losses.keys(), loss_values))))
      logger.info("Norm: {}".format(norm))
      if np.isnan(norm):
        import pdb;     pdb.set_trace()

      if i > 0 and i % 100 == 0:
        checkpoint_path = Path(training_path, "checkpoint/", "{}.ckpt".format(i))
        mkdirs_exists_ok(str(checkpoint_path.parent))
        save_path = saver.save(sess, str(checkpoint_path))
        logger.info("Saved model to {}".format(save_path))


if __name__ == "__main__":
  main(sys.argv)
