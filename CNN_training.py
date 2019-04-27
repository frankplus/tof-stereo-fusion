import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tables
from datetime import datetime
from dataset import *
from model import FusionCNN 


# load the imdb file
filepath = 'dataset.mat'
mat_content = tables.open_file(filepath)
training_data = mat_content.root.training_data[:]
training_label = mat_content.root.training_label[:]

validation_full_data = mat_content.root.validation_full_data[:]
validation_full_label = mat_content.root.validation_full_label[:]

test_data = mat_content.root.test_data[:]
test_label = mat_content.root.test_label[:]


train_dataset = DataSet(training_data,training_label, reshape=False)

validation_full_dataset = DataSet(validation_full_data,validation_full_label, reshape=False)

test_dataset = DataSet(test_data,test_label,reshape=False)

epoch_num = 1
batch_size = 4
learning_rate = 1e-4
n_filters = 8

current_time = datetime.now().strftime("%Y%m%d-%H%M")
checkpoints_dir = "checkpoints/{}".format(current_time)
#--------------------------------------------------------------------------
#------------------------------Graph Structure-----------------------------
#--------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, None, None, 2]) #input images
    label = tf.placeholder(tf.float32, shape=[None, None, None, 1]) #output fused disparity map
    mean_val_loss = tf.placeholder(tf.float32)

    cnn = FusionCNN(
                   n_filters = n_filters,
                   batch_size=batch_size,
                   learning_rate=learning_rate)

    loss, fused, summaries_training = cnn.model(x, label)

    optimizer = cnn.optimize(loss)

    summaries_val = cnn.validation_summary(x, label, mean_val_loss)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    val_writer = tf.summary.FileWriter(checkpoints_dir, graph,filename_suffix='_Val')

file = open('trainingStat_learningRate_'+str(learning_rate)+'_batchSize_'+str(batch_size)+'.txt','a')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True


with tf.Session(graph=graph,config=config) as sess:
  epoch_flag = 0 # remembers the epoch number of the previous step
  epoch_flag_val = 0
  train_mse_acc = 0
  step_counter = 0
  step_counter_glob = 0
 
  sess.run(tf.global_variables_initializer())
  while epoch_flag < epoch_num:
    batch = train_dataset.next_batch(batch_size)
    if epoch_flag < batch[2]: 
        # end of epoch: start validation and calculate mean squared error
        step_counter_val = 0
        validation_mse_acc = 0
        while epoch_flag_val < batch[2]:
            batch_val = validation_full_dataset.next_batch(1)
            validation_mse_acc += loss.eval(feed_dict={
                                            x: batch_val[0], label: batch_val[1]})
            step_counter_val += 1
            epoch_flag_val = batch_val[2]
        validation_mse = validation_mse_acc/step_counter_val
        train_mse = train_mse_acc / step_counter
        message = 'epoch %d training_mse %g validation_mse %g step_counter %d \n'%(batch[2], train_mse, validation_mse, step_counter)
        print(message)
        file.write(message)

        # write summaries
        [summaries_val_ev] = sess.run([summaries_val],feed_dict={x: batch_val[0],
                                                                label: batch_val[1],
                                                                mean_val_loss: validation_mse})
        val_writer.add_summary(summaries_val_ev, step_counter_glob)
        val_writer.flush()

        # start next epoch and train first batch
        step_counter = 1
        _, train_mse_acc, summary = sess.run([optimizer, loss, summaries_training],feed_dict={x: batch[0], label: batch[1]})
        epoch_flag = batch[2]
        train_writer.add_summary(summary, step_counter_glob)
    else:
        #train next batch
        _, loss_val, summary = sess.run([optimizer, loss, summaries_training],feed_dict={x: batch[0], label: batch[1]})
        step_counter += 1
        train_mse_acc += loss_val
        train_writer.add_summary(summary, step_counter_glob)
    step_counter_glob += 1
    train_writer.flush()

  # Save the variables to disk.
  save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step_counter_glob)
  print("Model saved in file: %s" % save_path)

  
  step_counter_test = 0
  test_mse_acc = 0
  epoch_flag_test = 0
  while epoch_flag_test < 1:
        batch_test = test_dataset.next_batch(1)
        test_mse_acc += loss.eval(feed_dict={x: batch_test[0], label: batch_test[1]})
        step_counter_test += 1
        epoch_flag_test = batch_test[2]
  test_mse = test_mse_acc/step_counter_test
  file.write('test error %d \n' % (test_mse))
  file.close()

"""
#-----------------------------------------------------------------------
#-----------------------------Show Results------------------------------
#-----------------------------------------------------------------------


  validation_data_shape = validation_full_data.shape
  n = validation_data_shape[0]
  canvas_orig = np.empty((validation_data_shape[0], validation_data_shape[1]))
  canvas_recon = np.empty((validation_data_shape[0], validation_data_shape[1]))
  for i in range(n):
        batch, batch_gt, _ = test_dataset.next_batch(1)#n
        # compute CNN output
        g = sess.run(fused, feed_dict={x_image: batch})

        tof_gt = np.zeros((validation_data_shape[1], validation_data_shape[2]))
        stereo_gt =  np.zeros((validation_data_shape[1], validation_data_shape[2]))
        tof_est =  np.zeros((validation_data_shape[1], validation_data_shape[2]))
        stereo_est = np.zeros((validation_data_shape[1], validation_data_shape[2]))
        # Display GT data
        tof_gt[:,:] = (batch_gt[0,:,:,0])
        stereo_gt[:,:] = (batch_gt[0,:,:,1])
        # Draw the reconstructed data
        tof_est[:,:] = (g[0,:,:,0])
        stereo_est[:,:] = (g[0,:,:,1])

        print("ToF Results")
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plot = plt.imshow(tof_gt, origin="upper", cmap="jet", clim=(-0.4, 0.4))
        plt.colorbar()
        plt.title(" ToF Error GT")

        plt.subplot(122)
        plt.imshow(tof_est, origin="upper", cmap="jet", clim=(-0.4, 0.4))
        plt.colorbar()
        plt.title("ToF Estimated Error")
        plt.show()

        print("Stereo Results")
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plot = plt.imshow(stereo_gt, origin="upper", cmap="jet", clim=(-0.4, 0.4))
        plt.colorbar()
        plt.title("Stereo Error GT")

        plt.subplot(122)
        plt.imshow(stereo_est, origin="upper", cmap="jet", clim=(-0.4, 0.4))
        plt.colorbar()
        plt.title("Stereo Estimated Error")
        plt.show()
"""
