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

fixed_folder = "/Users/surajshashidhar/git/image_registration_data/resampled"
moving_folder = "/Users/surajshashidhar/git/image_registration_data/resampled"

print("")
print(" ======== Starting preprocessing ==========")
ipr_obj = ipr.ImageProcessing(fixed_folder, moving_folder)
ipr_obj.getListofImages()
ipr_obj.readImagesfromList()
generator = ipr_obj.generateDataset(batch_size=2)

print(" ======== +++++ ==========")
print("")

with tf.device(device_name):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    tf.print(c)

    print("")
    print(" ======== Starting model building and training ==========")
    model_obj = model.Model(generator)
    model_obj.buildModel()
    model_obj.printModelSummary()
    model_obj.trainModel()
    model_obj.saveModelandWeights()
    print(" ========== Ending program =========")

