import tensorflow as tf
import functools


def define_scope(function, scope=None, *args, **kwargs):
  """args and kwargs are passed to variable_scope."""
  scope = scope or function.__name__
  vs_args = args
  vs_kwargs = kwargs
  @functools.wraps(function)
  def decorator(*args, **kwargs):
    with tf.variable_scope(scope, *vs_args, **vs_kwargs):
      return function(*args, **kwargs)
  return decorator

