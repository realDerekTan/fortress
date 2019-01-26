# I do not claim to own any of the facial recognition algorithm. Though it was modified, the essence of the code is still credited to Will Koehrsen.
# His original code that IsaacGAN was built upon can be found here: https://medium.com/@williamkoehrsen/facial-recognition-using-googles-convolutional-neural-network-5aa752b4240e

"""
Does not work at the moment! This is currently a work in progress, and it still contains many bugs and issues. We are currently working on
making it better, and hopefully we can get a fully working model sometime in the near future. But for now, here's our progress that we've
made so far.
"""

# Import useful Python libraries
import tensorflow as tf
import numpy as np
import os
import sys
import urllib
import tarfile
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import math
from collections import Counter
from scipy.misc import imresize
from nets import inception
from tensorflow.contrib import slim
from datetime import datetime
import time
import scipy

# URL for TensorFlow models
TF_MODELS_URL = "http://download.tensorflow.org/models/"

# Inception V3 CNN, built by Google
INCEPTION_V3_URL = TF_MODELS_URL + "inception_v3_2016_08_28.tar.gz"

# Directory to save model checkpoints
MODELS_DIR = "models/cnn"
INCEPTION_V3_CKPT_PATH = MODELS_DIR + "/inception_v3.ckpt"

# define function since datasets don't work on my computer
def download_and_uncompress_tarball(tarball_url, dataset_dir):
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

# Make the model directory if it does not exist
if not tf.gfile.Exists(MODELS_DIR):
 tf.gfile.MakeDirs(MODELS_DIR)

# Download the appropriate model of Inception
if not os.path.exists(INCEPTION_V3_CKPT_PATH):
    download_and_uncompress_tarball(INCEPTION_V3_URL, MODELS_DIR)

# Full deep-funneled images dataset
FACES_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
IMAGES_DOWNLOAD_DIRECTORY = "tmp/faces"
IMAGES_DIRECTORY = "images/faces"

if not os.path.exists(IMAGES_DOWNLOAD_DIRECTORY):
 os.makedirs(IMAGES_DOWNLOAD_DIRECTORY)

# If the file has not already been downloaded, retrieve and extract it
if not os.path.exists(IMAGES_DOWNLOAD_DIRECTORY + "/lfw-deepfunneled.tgz"):
 download_and_uncompress_tarball(FACES_URL, IMAGES_DOWNLOAD_DIRECTORY)

# Count number of photos of each individual
people_number = []

people = IMAGES_DOWNLOAD_DIRECTORY + "/lfw-deepfunneled"

for person in people:
 folder_path = os.path.join(people, person)
 num_images = len([f for f in os.listdir(folder_path) if os.path.isfile(f)])
 people_number.append((person, num_images))

# Sort the list of tuples by the number of images
people_number = sorted(people_number, key=lambda x: x[1], reverse=True)

# Determine number of people with one image
people_with_one_photo = [(person) for person, num_images in people_number if num_images==1]
print("Individuals with one photo: {}".format(len(people_with_one_photo)))

# Number of individuals to retain
num_classes = 10

# Make the images directory if it does not exist
if not os.path.exists(IMAGES_DIRECTORY):
 os.mkdir(IMAGES_DIRECTORY)

 # Take the ten folders with the most images and move to new directory
 # Rename the folders with the number of images and name of individual
 for person in people_number[:num_classes]:
 name = person[0]
 # Original download directory path
 folder_path = IMAGES_DOWNLOAD_DIRECTORY + '/lfw-deepfunneled/' + name
 formatted_num_images = str(person[1]).zfill(3)
 new_folder_name = "{} {}".format(formatted_num_images, name)
 image_new_name = IMAGES_DIRECTORY + "/" + new_folder_name

 # Make a new folder for each individual in the images directory
 os.mkdir(IMAGES_DIRECTORY + '/' + name)
 # Copy the folder from the download location to the new folder
 copy_tree(folder_path, IMAGES_DIRECTORY + '/' + name)
 # Rename the folder with images and individual
 os.rename(IMAGES_DIRECTORY + '/' + name, image_new_name)

# Map each class to an integer label
class_mapping = {}
class_images = {}
# Create dictionary to map integer labels to individuals
# Class_images will record number of images for each class
for index, directory in enumerate(os.listdir("images/faces")):
 class_mapping[index] = directory.split(" ")[1]
 class_images[index] = int(directory.split(' ')[0])
print(class_mapping)

total_num_images = np.sum(list(class_images.values()))
print("Individual \t Composition of Dataset\n")
for label, num_images in class_images.items():
 print("{:20} {:.2f}%".format(
 class_mapping[label], (num_images / total_num_images) * 100))

image_arrays = []
image_labels = []
root_image_directory = "images/faces/"
for label, person in class_mapping.items():
 for directory in os.listdir(root_image_directory):
 if directory.split(" ")[1] == person:
 image_directory = root_image_directory + directory
 break

 for image in os.listdir(image_directory):
 image = plt.imread(os.path.join(image_directory, image))
 image_arrays.append(image)
 image_labels.append(label)
image_arrays = np.array(image_arrays)
image_labels = np.array(image_labels)
print(image_arrays.shape, image_labels.shape)

# Fractions for each dataset
train_frac = 0.70
valid_frac = 0.05
test_frac = 0.25

# This function takes in np arrays of images and labels along with split fractions
# and returns the six data arrays corresponding to each dataset as the appropriate type
def create_data_splits(X, y, train_frac=train_frac, test_frac=test_frac, valid_frac=valid_frac):
 X = np.array(X)
 y = np.array(y)

 # Make sure that the fractions sum to 1.0
 assert (test_frac + valid_frac + train_frac == 1.0), "Test + Valid + Train Fractions must sum to 1.0"

 X_raw_test = []
 X_raw_valid = []
 X_raw_train = []

 y_raw_test = []
 y_raw_valid = []
 y_raw_train = []

 # Randomly order the data and labels
 random_indices = np.random.permutation(len(X))
 X = X[random_indices]
 y = y[random_indices]

 for image, label in zip(X, y):

 # Number of images that correspond to desired fraction
 test_length = math.floor(test_frac * class_images[label])
 valid_length = math.floor(valid_frac * class_images[label])

 # Check to see if the datasets have the right number of labels (and images)
 if Counter(y_raw_test)[label] < test_length:
 X_raw_test.append(image)
 y_raw_test.append(label)
 elif Counter(y_raw_valid)[label] < valid_length:
 X_raw_valid.append(image)
 y_raw_valid.append(label)
 else:
 X_raw_train.append(image)
 y_raw_train.append(label)

 return np.array(X_raw_train, dtype=np.float32), np.array(X_raw_valid, dtype=np.float32), np.array(X_raw_test, dtype=np.float32), np.array(y_raw_train, dtype=np.int32), np.array(y_raw_valid, dtype=np.int32), np.array(y_raw_test, dtype=np.int32)
# Create all the testing splits using the create_splits function
X_train, X_valid, X_test, y_train, y_valid, y_test = create_data_splits(image_arrays, image_labels)

# Check the number of images in each dataset split
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)

# Import matplotlib and use magic command to plot in notebook
%matplotlib inline
# Function to plot an array of RGB values
def plot_color_image(image):
 plt.figure(figsize=(4,4))
 plt.imshow(image.astype(np.uint8), interpolation='nearest')
 plt.axis('off')
import random
# PLot 2 examples from each class
num_examples = 2
# Iterate through the classes and plot 2 images from each
for class_number, person in class_mapping.items():
 print('{} Number of Images: {}'.format(person, class_images[class_number]))
 example_images = []
 while len(example_images) < num_examples:
 random_index = np.random.randint(len(X_train))
 if y_train[random_index] == class_number:
 example_images.append(X_train[random_index])

 for i, image in enumerate(example_images):
 plt.subplot(100 + num_examples*10 + i + 1)
 plt.imshow(image.astype(np.uint8), interpolation='nearest')
 plt.axis('off')
 plt.show()

# Function takes in an image array and returns the resized and normalized array
def prepare_image(image, target_height=299, target_width=299):
 image = imresize(image, (target_width, target_height))
 return image.astype(np.float32) / 255

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 299, 299, 3], name='X')
is_training = tf.placeholder_with_default(False, [])
# Run inception function to determine endpoints
with slim.arg_scope(inception.inception_v3_arg_scope()):
 logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=is_training)
# Create saver of network before alterations
inception_saver = tf.train.Saver()
print(end_points)

# Isolate the trainable layer
prelogits = tf.squeeze(end_points['PreLogits'], axis=[1,2])
# Define the training layer and the new output layer
n_outputs = len(class_mapping)
with tf.name_scope("new_output_layer"):
 people_logits = tf.layers.dense(prelogits, n_outputs, name="people_logits")
 probability = tf.nn.softmax(people_logits, name='probability')
# Placeholder for labels
y = tf.placeholder(tf.int32, None)
# Loss function and training operation
# The training operation is passed the variables to train which includes only the single layer
with tf.name_scope("train"):
 xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=people_logits, labels=y)
 loss = tf.reduce_mean(xentropy)
 optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
 # Single layer to be trained
 train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="people_logits")
 # The variables to train are passed to the training operation
 training_op = optimizer.minimize(loss, var_list=train_vars)

# Accuracy for network evaluation
with tf.name_scope("eval"):
 correct = tf.nn.in_top_k(predictions=people_logits, targets=y, k=1)
 accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Intialization function and saver
with tf.name_scope("init_and_saver"):
 init = tf.global_variables_initializer()
 saver = tf.train.Saver()

# Function takes in an array of images and labels and processes the images to create
# a batch of a given size
def create_batch(X, y, start_index=0, batch_size=4):

 stop_index = start_index + batch_size
 prepared_images = []
 labels = []

 for index in range(start_index, stop_index):
 prepared_images.append(prepare_image(X[index]))
 labels.append(y[index])

 # Combine the images into a single array by joining along the 0th axis
 X_batch = np.stack(prepared_images)
 # Combine the labels into a single array
 y_batch = np.array(labels, dtype=np.int32)

 return X_batch, y_batch

X_valid, y_valid = create_batch(X_valid, y_valid, 0, len(X_valid))
print(X_valid.shape, y_valid.shape)

with tf.name_scope("tensorboard_writing"):
 # Track validation accuracy and loss and training accuracy
 valid_acc_summary = tf.summary.scalar(name='valid_acc', tensor=accuracy)
 valid_loss_summary = tf.summary.scalar(name='valid_loss', tensor=loss)
 train_acc_summary = tf.summary.scalar(name='train_acc', tensor=accuracy)
# Merge the validation stats
 valid_merged_summary = tf.summary.merge(inputs=[valid_acc_summary, valid_loss_summary])
# Specify the directory for the FileWriter
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "{}_unaugmented".format(now)
logdir = "tensorboard/faces/" + model_dir
file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())

# start training model
n_epochs = 100
batch_size = 32
# Early stopping parameters
max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.float("inf")
# Show progress every show_progress epochs
show_progress = 1
# Want to iterate through the entire training set every epoch
n_iterations_per_epoch = len(X_train) // batch_size
# Specify the directory for the FileWriter
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "{}_unaugmented".format(now)
logdir = "tensorboard/faces/" + model_dir
file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
# This is the pre-trained model checkpoint training path
inception_v3_checkpoint_path = "models/cnn/inception_v3.ckpt"
# This is the checkpoint path for our trained model with no dataaugmentation
unaugmented_training_path = "models/cnn/inception_v3_faces_unaugmented.ckpt"
with tf.Session() as sess:
 init.run()
 # Restore all the weights from the original CNN
 inception_saver.restore(sess, inception_v3_checkpoint_path)

 t0 = time.time()
 for epoch in range(n_epochs):
 start_index = 0
 # Each epoch, iterate through all the training instances
 for iteration in range(n_iterations_per_epoch):
 X_batch, y_batch = create_batch(X_train, y_train, start_index, batch_size)
 # Train the trainable layer
 sess.run(training_op, {X: X_batch, y: y_batch})
 start_index += batch_size

 # Display the progress of training and write to the TensorBoard directory
 # for later visualization of the training
 if epoch % show_progress == 0:
 train_summary = sess.run(train_acc_summary, {X: X_batch, y: y_batch})
 file_writer.add_summary(train_summary, (epoch+1))
 #Size for validation limited by GPU memory (68 images will work)
 valid_loss, valid_acc, valid_summary = sess.run([loss, accuracy, valid_merged_summary], {X: X_valid, y: y_valid})
 file_writer.add_summary(valid_summary, (epoch+1))
 print('Epoch: {:4} Validation Loss: {:.4f} Accuracy: {:4f}'.format(
 epoch+1, valid_loss, valid_acc))

 # Check to see if network is still improving, if improved during epoch
 # a snapshot of the model will be saved to retain the best model
 if valid_loss < best_loss:
 best_loss = valid_loss
 checks_without_progess = 0
 save_path = saver.save(sess, unaugmented_training_path)

 # If network is not improving for a specified number of epochs, stop training
 else:
 checks_without_progress += 1
 if checks_without_progress > max_checks_without_progress:
 print('Stopping Early! Loss has not improved in {} epochs'.format(
 max_checks_without_progress))
 break

 t1 = time.time()

print('Total Training Time: {:.2f} minutes'.format( (t1-t0) / 60))

# Evaluate training model
eval_batch_size = 32
n_iterations = len(X_test) // eval_batch_size
with tf.Session() as sess:
 # Restore the new trained model
 saver.restore(sess, unaugmented_training_path)

 start_index = 0
 # Create a dictionary to store all the accuracies
 test_acc = {}

 t0 = time.time()
 # Iterate through entire testing set one batch at a time
 for iteration in range(n_iterations):
 X_test_batch, y_test_batch = create_batch(X_test, y_test, start_index, batch_size=eval_batch_size)
 test_acc[iteration] = accuracy.eval({X: X_test_batch, y:y_test_batch})
 start_index += eval_batch_size
print('Iteration: {} Batch Testing Accuracy: {:.2f}%'.format(
 iteration+1, test_acc[iteration] * 100))

 t1 = time.time()

 # Final accuracy is mean of each batch accuracy
 print('\nFinal Testing Accuracy: {:.4f}% on {} instances.'.format(
 np.mean(list(test_acc.values())) * 100, len(X_test)))
 print('Total evaluation time: {:.4f} seconds'.format((t1-t0)))

# Take in an image as an array and return image with a [dx, dy] shift
def shift_image(image_array, shift):
 return scipy.ndimage.interpolation.shift(image_array, shift, cval=0)
# Four shifts of 30 pixels
shifts = [[30,0], [-30,0], [0, 30], [0,-30]]
shifted_images = []
shifted_labels = []
# Iterate through all training images
for image, label in zip(X_train, y_train):

 # Swap the color channel and height axis
 layers = np.swapaxes(image, 0, 2)

 # Apply four shifts to each original image
 for shift in shifts:
 transposed_image_layers = []

 # Apply the shift to the image one layer at a time
 # Each layer is an RGB color channel
 for layer in layers:
 transposed_image_layers.append(shift_image(layer, shift))

 # Stack the RGB layers to get one image and reswap the axes
 transposed_image = np.stack(transposed_image_layers)
 transposed_image = np.swapaxes(transposed_image, 0, 2)

 # Add the shifted images and the labels to a list
 shifted_images.append(transposed_image)
 shifted_labels.append(label)
# Convert the images and labels to numpy arrays
shifted_images = np.array(shifted_images)
shifted_labels = np.array(shifted_labels)
print(shifted_images.shape,shifted_labels.shape)

ex_index = 5
# Plot original image
plot_color_image(X_train[ex_index])
plt.title("Original Image of {}".format(class_mapping[y_train[ex_index]]))
plt.show()
ex_shifted_images = shifted_images[ex_index*4:(ex_index*4)+ 4]
# Plot four shifted images
for i, image in enumerate(ex_shifted_images):
 shift = shifts[i]
 plt.subplot(2,2,i+1)
 plt.imshow(image.astype(np.uint8), interpolation='nearest')
 plt.title('Shift: {}'.format(shift))
 plt.axis('off')
plt.show()

# Create a new training set with the original and shifted images
X_train_exp = np.concatenate((shifted_images, X_train))
y_train_exp = np.concatenate((shifted_labels, y_train))
print(X_train_exp.shape, y_train_exp.shape)

restart_augmented_training_path = "models/cnn/inception_v3_faces_restart_augmented.ckpt"
with tf.Session() as sess:
 init.run()
 inception_saver.restore(sess, unaugmented_training_path)

images_flipped = []
labels_flipped = []
# Flip every image in the training set
for image, label in zip(X_train, y_train):
 images_flipped.append(np.fliplr(image))
 labels_flipped.append(label)
# Convert the flipped images and labels to arrays
images_flipped = np.array(images_flipped)
labels_flipped = np.array(labels_flipped)

ex_index = 652
plot_color_image(X_train[ex_index])
plt.title('Original Image of {}'.format(class_mapping[y_train[ex_index]]))
plt.show()
plot_color_image(images_flipped[ex_index])
plt.title('Left to Right Flipped Image')
plt.show()

X_train_with_flip = np.concatenate((X_train, images_flipped))
y_train_with_flip = np.concatenate((y_train, labels_flipped))
print(X_train_with_flip.shape, y_train_with_flip.shape)

# Visualize performance of CNN
def classify_image(index, images=X_test, labels=y_test):
 image_array = images[index]
 label = class_mapping[labels[index]]

 prepared_image = prepare_image(image_array)
 prepared_image = np.reshape(prepared_image, newshape=(-1, 299, 299, 3))

 with tf.Session() as sess:
 saver.restore(sess, restart_augmented_training_path)
 predictions = sess.run(probability, {X: prepared_image})

 predictions = [(i, prediction) for i, prediction in enumerate(predictions[0])]
 predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
 print('\nCorrect Answer: {}'.format(label))
 print('\nPredictions:')
 for prediction in predictions:
 class_label = prediction[0]
 probability_value = prediction[1]
 label = class_mapping[class_label]
 print("{:26}: {:.2f}%".format(label, probability_value * 100))

 plot_color_image(image_array)
 return predictions


# this is to test specific images
predictions = classify_image(replace with number)


eval_batch_size = 32
n_iterations = len(X_test) // eval_batch_size
# Evaluation with saved predictions
with tf.Session() as sess:
 saver.restore(sess, restart_augmented_training_path)

 test_predictions = []
 start_index = 0

 # Add each set of predictions to a list
 for iteration in range(n_iterations):
 X_test_batch, y_test_batch = create_batch(X_test, y_test, start_index, batch_size=eval_batch_size)
 test_predictions.append(probability.eval({X: X_test_batch, y: y_test_batch}))
 start_index += eval_batch_size
# Convert list of predictions to np array
test_predictions = np.array(test_predictions)
test_predictions.shape

# Reshape predictions to correct shape and generate label array
test_predictions = np.reshape(test_predictions, (-1, 10))
test_predictions_label = np.argmax(test_predictions, axis=1)
test_predictions_label.shape

# A few of the testing examples are left off by the batching process
y_test_eval = y_test[:352]
# Make sure that accuracy agrees with earlier evaluation
test_accuracy = np.mean(np.equal(test_predictions_label, y_test_eval))
print("Test Accuracy: {:.2f}%".format(test_accuracy*100))

print('Individual: \t Composition of Test Set \t Composition of Predictions')
total_test_images = len(y_test_eval)
for label, individual in class_mapping.items():
 n_test_images = Counter(y_test_eval)[label]
 n_predictions = Counter(test_predictions_label)[label]
 print("{:26} {:5.2f}% {:26.2f}%".format(
 individual, n_test_images / total_test_images * 100, n_predictions / total_test_images * 100))

# Visualize distribution of images in test set
bins = np.arange(11)
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(wspace=0.25)
plt.subplot(1,2,1)
plt.hist(y_test_eval, bins=bins, lw=1.2, edgecolor='black')
plt.title('Frequency of Classes in Test Set'); plt.xlabel('Class'); plt.ylabel('Count');
names = list(class_mapping.values())
xlabels = [name.split("_")[1] for name in names]
plt.xticks(bins+0.5, xlabels, rotation='vertical');
plt.subplot(1,2,2)
# Visualize distribution of images in predictions
bins = np.arange(11)
plt.hist(test_predictions_label, bins=bins, lw=1.2, edgecolor='black')
plt.title('Frequency of Classes in Predictions'); plt.xlabel('Class'); plt.ylabel('Count');
names = list(class_mapping.values())
xlabels = [name.split("_")[1] for name in names]
plt.xticks(bins+0.5, xlabels, rotation='vertical');

relative_differences = []
for label, individual in class_mapping.items():
 n_test_images = Counter(y_test_eval)[label]
 n_predictions = Counter(test_predictions_label)[label]
 n_test_pct = n_test_images / total_test_images
 n_predictions_pct = n_predictions / total_test_images

 relative_differences.append((n_predictions_pct - n_test_pct) / n_test_pct * 100)
colors = ['orange' if difference < 0.0 else 'blue' for difference in relative_differences]
# Visualize relative difference in
bins = np.arange(10)
fig, ax = plt.subplots()
plt.bar(bins, relative_differences, lw=1.2, color=colors, edgecolor='black')
plt.title('Relative Differences between Test and Prediction Composition'); plt.xlabel('Class'); plt.ylabel('Relative Difference (%)');
names = list(class_mapping.values())
xlabels = [name.split("_")[1] for name in names]
plt.xticks(bins, xlabels, rotation='vertical');
