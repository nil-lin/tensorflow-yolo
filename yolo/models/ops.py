from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha, dtype=tf.float32):
  """leaky relu 
  if x > 0:
    return x
  else:
    return alpha * x
  Args:
    x : Tensor
    alpha: float
  Return:
    y : Tensor
  """
  x = tf.cast(x, dtype=dtype)
  bool_mask = (x > 0)
  mask = tf.cast(bool_mask, dtype=dtype)
  return 1.0 * mask * x + alpha * (1 - mask) * x
