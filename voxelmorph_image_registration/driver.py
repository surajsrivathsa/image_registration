import os, sys
import time
import numpy as np
import tensorflow as tf
import logging
assert tf.__version__.startswith('2.')
import neurite as ne
import nibabel as nb
import argparse
import nibabel.processing as nbp
import voxelmorph as vxm
import re
import glob
import image_processing as ipr
import model


print("Tensorflow version: {}".format(tf.__version__))
print("Nibabel version: {}".format(nb.__version__))


gpus = tf.config.experimental.list_physical_devices('GPU')
device_name = None

if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[6], 'GPU')
    logical_gpus = gpus[6]
    print(len(gpus), "Physical GPUs,", len(logical_gpus.name), "Logical GPU")
    print("Selected GPU: {}".format(logical_gpus))
    device_name = re.sub("/physical_device:", "", gpus[6].name)
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

else:
    cpus = tf.config.experimental.list_physical_devices('CPU')
    print("Using CPU as GPU is unavcailable: {}".format(cpus[0].name))
    device_name = re.sub("/physical_device:", "", cpus[0].name)

fixed_folder = "/project/shashidh/Voxel_morph_dataset"
moving_folder = "/project/shashidh/Voxel_morph_dataset"
model_save_path = "/project/shashidh/t1_mv_t1_fix_registration_model"
os.mkdir(model_save_path)
model_weights_save_path = "/project/shashidh/t1_mv_t1_fix_registration_weights"
num_bins = 10
bin_centers = np.linspace(0, 0.7, num_bins*2+1)[1::2]

print()
#print(" =========== Choose Mutual Info loss and printing bin centers ==============")
#print(bin_centers)
#print("========== ================= =============")
print(" ============ Using Normalized Cross Corelation ============")
print()

#similarity_loss_type = vxm.losses.NMI( bin_centers = bin_centers, vol_size = (128, 128, 128)).loss
similarity_loss_type = vxm.losses.NCC().loss
similarity_loss_weight = 1
regularizer_loss_type = vxm.losses.Grad("l2").loss
regularizer_loss_weight = 0.2
epochs = 100
batch_size = 3
steps_per_epoch = 50//3
vol_shape = (128, 128, 128)

print("Epochs, Batchsize, steps per epoch and volume shape are {}, {}, {} and {}".format(epochs, batch_size, steps_per_epoch, vol_shape))

print("")
print(" ======== Starting preprocessing ==========")
ipr_obj = ipr.ImageProcessing(fixed_folder, moving_folder)
ipr_obj.getListofImages()
ipr_obj.readImagesfromListnew()
generator = ipr_obj.generateDataset(batch_size=batch_size)

print(" ======== +++++ ==========")
print("")

with tf.device(device_name):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    tf.print(c)

    print("")
    print(" ======== Starting model building and training ==========")
    model_obj = model.Model(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, vol_shape=vol_shape, model_save_path=model_save_path, model_weights_save_path=model_weights_save_path, similarity_loss_type=similarity_loss_type, similarity_loss_weight= similarity_loss_weight, regularizer_loss_type = regularizer_loss_type, regularizer_loss_weight = regularizer_loss_weight )
    model_obj.buildModel()
    model_obj.printModelSummary()
    model_obj.trainModel()
    #model_obj.saveModelandWeights()
    print(" ========== Ending program =========")


