import tensorflow as tf
import ops
#import utils
#from reader import Reader
from net import Net

REAL_LABEL = 0.9

class FusionCNN:
  def __init__(self,
               n_filters = 32,
               batch_size=8,
               norm=None,
               learning_rate=2e-4,
               beta1=0.5,
              ):
    """
    Args:
      n_filters: integer, number of filters for each convolutional layer
      batch_size: integer, batch size
      norm: 'None, 'instance' or 'batch'
      learning_rate: float, learning rate for Adam
      beta1: float, momentum term of Adam
    """
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.n_filters = n_filters
    self.norm = norm

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.net = Net('net', n_filters=self.n_filters, is_training=self.is_training, norm=self.norm)


  def model(self, x, label):
    """
    Args:
      x: ensor of float, sample of the validation set
      label: tensor of float, label related to "x"
    Output:
      loss: float, loss value for the current batch
      fused: tensor of float, output of the CNN for the input "x"
      :return: summaries_training: tf.summary for the current training step
    """
  
    fused = self.net(x)
    print(fused)
    print(label)
    loss = self.l2_loss(fused, label)

    # summary
    tf.summary.scalar('loss/l2', loss)

    summaries_training_list = [
                               tf.summary.scalar('training_error',loss)
                              ]
    summaries_training = tf.summary.merge(summaries_training_list)

    return loss, fused, summaries_training

  def validation_summary(self, x, label, mean_loss):
    """
    Args:
      x: tensor of float, sample of the validation set
      label: tensor of float, label related to "x"
      mean_loss: float, validation error in the current epoch
    Output:
      validation_summary: tf.summary, summury for the validation set, images for a sample of the set is saved
    """
    fused = self.net(x)

    x_shape = tf.shape(x)
    x_tof = tf.slice(x, [0, 0, 0, 1], [1, x_shape[1], x_shape[2], 1])
    x_stereo = tf.slice(x, [0, 0, 0, 0], [1, x_shape[1], x_shape[2], 1])
    label_toShow = tf.slice(label, [0, 0, 0, 0], [1, x_shape[1], x_shape[2], 1])
    fused_toShow = tf.slice(fused, [0, 0, 0, 0], [1, x_shape[1], x_shape[2], 1])
    summaries_val_list = [tf.summary.image('val/input_tof', x_tof),
                               tf.summary.image('val/input_stereo', x_stereo),
                               tf.summary.image('val/label', label_toShow),
                               tf.summary.image('val/fused', fused_toShow),

                               tf.summary.scalar('validation_error', mean_loss)
                               ]
    summaries_val = tf.summary.merge(summaries_val_list)

    return summaries_val

  def optimize(self, loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with the given learning rate [COMMENTED: for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps]
      """
      
      learning_rate = self.learning_rate
 
      beta1 = self.beta1
      
      global_step = tf.Variable(0, trainable=False)
      """
	  end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
	  learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )"""
      #tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    Net_optimizer = make_optimizer(loss, self.net.variables, name='Adam_net')

    with tf.control_dependencies([Net_optimizer]):
      return tf.no_op(name='optimizers')


  def l2_loss(self,output, label):
    """ loss (MSE norm)
    """
    loss = tf.reduce_mean(tf.squared_difference(label,output))
    return loss
	
  def l1_loss(self,output, label):
    """ loss (MAE norm)
    """
    loss = tf.reduce_mean(tf.absolute_difference(label,output))
    return loss
