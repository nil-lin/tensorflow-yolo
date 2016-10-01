from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def leaky_relu(x, alpha):
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
  bool_mask = (x > 0)
  mask = tf.cast(bool_mast, dtype=np.float32)
  return 1.0 * mask * x + alpha * (1 - mask) * x