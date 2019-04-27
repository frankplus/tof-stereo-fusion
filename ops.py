import tensorflow as tf

### Generator layers
def conv_layer(input, w_size, n_filters, reuse=False, norm='instance', activation='relu', is_training=True, name='conv'):
  """ A w_size x w_size Convolution-ReLU layer with n_filters filters and stride 1
  Args:
    input: 4D tensor
	w_size: integer, window size of the filter (w_size x w_size)
    n_filters: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'conv'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[w_size, w_size, input.get_shape()[3], n_filters])
    h_w_size = int(w_size/2);
    padded = tf.pad(input, [[0,0],[h_w_size,h_w_size],[h_w_size,h_w_size],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

def last_conv_layer(input, w_size, n_filters, reuse=False, name='conv'):
  """ A w_size x w_size Convolution layer with n_filters filters and stride 1
  Args:
    input: 4D tensor
	w_size: integer, window size of the filter (w_size x w_size)
    n_filters: integer, number of filters (output depth)
    name: string, e.g. 'conv'
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[w_size, w_size, input.get_shape()[3], n_filters])
    h_w_size = int(w_size/2);
    padded = tf.pad(input, [[0,0],[h_w_size,h_w_size],[h_w_size,h_w_size],[0,0]], 'REFLECT')
    output = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    return output



def t_conv_layer(input, w_size, n_filters, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
  """ A w_size x w_size fractional-strided-Convolution-BatchNorm-ReLU layer
      with n_filters filters, stride 1/2
  Args:
    input: 4D tensor
    w_size: integer, window size of the filter (w_size x w_size)
    n_filters: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 't_conv'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",
      shape=[w_size, w_size, n_filters, input_shape[3]])

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, n_filters]
    fsconv = tf.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(fsconv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output


### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_training, norm='instance'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope("batch_norm"):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)
