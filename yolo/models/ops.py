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

def iou(boxs1, boxs2):
    """calculate the iou of boxs1 and boxs2
    Args:
        boxs1: 2-D tensor, shape=[m, 4] (xmin, ymin, xmax, ymax)
        boxs2: 2-D tensor, shape=[n, 4] (xmin, ymin, xmax, ymax)
    Return:
        IOU: 2-D tensor (m, n)
    """
    m = tf.shape(boxs1)[0]
    n = tf.shape(boxs2)[0]
    
    extend_boxs1 = tf.tile(boxs1, (1, n))
    extend_boxs1 = tf.reshape(extend_boxs1, (m, n, 4))
    
    extend_boxs2 = tf.tile(boxs2, (m, 1))
    extend_boxs2 = tf.reshape(extend_boxs2, (m, n, 4))
    
    boxs = tf.pack([extend_boxs1, extend_boxs2])
    
    lr = tf.maximum(boxs[0, :, :, 0:2], boxs[1, :, :, 0:2])
    
    rd = tf.minimum(boxs[0, :, :, 2:], boxs[1, :, :, 2:])
    
    intersection = rd - lr
    inter_square = intersection[:, :, 0] * intersection[:, :, 1]
    
    mask = tf.cast(intersection[:, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxs1[:, 2] - boxs1[:, 0]) * (boxs1[:, 3] - boxs1[:, 1])
    square2 = (boxs2[:, 2] - boxs2[:, 0]) * (boxs2[:, 3] - boxs2[:, 1])
    
    square1 = tf.reshape(tf.tile(square1, (n,)), (m, n))
    square2 = tf.reshape(tf.tile(square2, (m,)), (m, n))
    
    return inter_square/(square1 + square2 - inter_square)
