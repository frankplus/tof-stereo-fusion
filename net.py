import tensorflow as tf
import ops
#import utils

class Net:
  def __init__(self, name, n_filters, is_training, norm=None):
    self.name = name
    self.n_filters = n_filters
    self.reuse = False
    self.norm = norm
    self.is_training = is_training

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x n_channels
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      conv1 = ops.conv_layer(input, 3, self.n_filters, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='conv1')
      conv2 = ops.conv_layer(conv1, 3, self.n_filters, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='conv2')
      conv3 = ops.conv_layer(conv2, 3, self.n_filters, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='conv3')
      conv4 = ops.conv_layer(conv3, 3, self.n_filters, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='conv4')
      
      output = ops.last_conv_layer(conv4, 3, 1, reuse=self.reuse, name='output')
 
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    output = self.__call__(input)
    #image = utils.batch_convert2int(self.__call__(input))
    #image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return tf.squeeze(output)
