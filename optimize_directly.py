#!/usr/bin/env python3
import sys
import soundfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tf_visqol import TFVisqol, _DTYPE
from util import resample
from util import opus_transcode


def get_loss(ref_var, deg_var, fs, n_samples):
  tf_visqol = TFVisqol(fs)
  nsim = tf_visqol.visqol(ref_var, deg_var, n_samples)
  sq_loss = tf.cast(tf.reduce_mean(tf.squared_difference(deg_var, ref_var)), _DTYPE)
  loss = 0.1*tf.log(sq_loss) - nsim
  minimize_op = tf.train.AdamOptimizer().minimize(loss)
  return loss, minimize_op

def main(argv):
  original, original_fs = soundfile.read("original.wav")
  original_mono = original[:, 0] if len(original.shape) > 1 else original

  fs = 16000
  reference = resample(original_mono, original_fs, fs)

  reference_path = "original_16k_mono.wav"
  soundfile.write(reference_path, reference, samplerate=fs, format="wav", subtype="float")
  opus_path = "opus_16k_mono.wav"
  opus_transcode(reference_path, opus_path)

  degraded, _ = soundfile.read(opus_path)

  n = reference.size
  ref_var = tf.placeholder(_DTYPE, (1, n), name="ref")
  deg_var = tf.get_variable(
    "deg_input",
    dtype=_DTYPE,
    initializer=degraded.astype(np.float32, copy=False).reshape(1, -1))

  loss_var, minimize_op = get_loss(ref_var, deg_var, fs, n)
  model = tf.global_variables_initializer()
  init_op = tf.initialize_all_variables()
  feed_dict = {ref_var: reference.reshape(1, -1)}

  with tf.Session() as sess:
    # Initialize model.
    sess.run(init_op)
    sess.run(model)

    for i in tqdm(range(1000)):
      _, loss = sess.run([minimize_op, loss_var], feed_dict)
      print(loss)

      if i > 0 and i % 100 == 0:
        try:
          deg = sess.run(deg_var)
          soundfile.write("./test_deg_{}.wav".format(i), deg.astype(np.float32, copy=False).T, fs, 'float')
        except Exception as e:
          import pdb;   pdb.set_trace()
          print("")

if __name__ == "__main__":
  main(sys.argv)
