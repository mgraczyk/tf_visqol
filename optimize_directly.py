#!/usr/bin/env python3
import sys
import soundfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tf_visqol import TFVisqol, _DTYPE
from util import resample

# Unused. FFT is not implemented on the CPU.
def is_distance(ref_var, deg_var):
  ref_spec = tf.abs(tf.fft(tf.complex(tf.to_float(ref_var), imag=tf.constant(0, dtype=tf.float32))))
  deg_spec = tf.abs(tf.fft(tf.complex(tf.to_float(deg_var), imag=tf.constant(0, dtype=tf.float32))))
  ratio = ref_spec / deg_spec
  is_distance = tf.reduce_sum(ratio - tf.log(ratio) - 1, axis=[1])
  return tf.to_double(is_distance)

def get_loss(ref_var, deg_var, fs):
  tf_visqol = TFVisqol(fs)
  nsim = tf_visqol.visqol(ref_var, deg_var)
  sq_loss = tf.cast(tf.reduce_mean(tf.squared_difference(deg_var, ref_var)), _DTYPE)
  loss = 0.1*tf.log(sq_loss) - nsim
  minimize_op = tf.train.AdamOptimizer().minimize(loss)
  return loss, minimize_op

def main(argv):
  original, fs_old = soundfile.read("original.wav")
  opus_output, _ = soundfile.read("low.wav")

  fs = 16000
  original = resample(original, fs_old, fs)
  original = original[:, 0].astype(np.float32, copy=False).reshape(1, -1)
  soundfile.write("original_16k_mono.wav", original.astype(np.float32, copy=False).T, fs, "float")

  opus_output = resample(opus_output, fs_old, fs)
  opus_output = opus_output[:, 0].astype(np.float32, copy=False).reshape(1, -1)
  soundfile.write("opus_16k_mono.wav", opus_output.astype(np.float32, copy=False).T, fs, "float")

  n = original.size
  ref_var = tf.placeholder(_DTYPE, (1, n), name="ref")
  deg_var = tf.get_variable(
    "deg_input",
    dtype=_DTYPE,
    initializer=opus_output)

  loss_var, minimize_op = get_loss(ref_var, deg_var, fs)
  model = tf.global_variables_initializer()
  init_op = tf.initialize_all_variables()

  with tf.Session() as sess:
    # Initialize model.
    sess.run(init_op)
    sess.run(model)

    for i in tqdm(range(1000)):
      feed_dict = {ref_var: original.reshape(1, -1)}
      _, loss = sess.run([minimize_op, loss_var], feed_dict)
      print(loss)

      if i > 0 and i % 100 == 0:
        try:
          deg = sess.run(deg_var)
          soundfile.write("./test_deg_{}.wav".format(i), deg.astype(np.float32, copy=False).T, fs, 'float')
        except Exception as e:
          import pdb; pdb.set_trace()
          print("")

if __name__ == "__main__":
  main(sys.argv)
