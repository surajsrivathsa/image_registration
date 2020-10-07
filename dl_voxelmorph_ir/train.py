import os
import sys
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


def read_subject_list(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    return [x.strip() for x in content]


def configure_device(device_id):
    if device_id is not None and device_id != '-1':
        device = '/gpu:' + device_id
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    else:
        device = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return device


def train_generator():

    # dimension-specific utilities
    prep = lambda v: v[:, 48:-48, 31:-33, 3:-29, :].astype('float32') / 255.0
    norm_filename = 'norm_talairach.mgz'

    # create base scan-to-scan generator
    subjs = read_subject_list('/autofs/cluster/vxmdata1/FS_Slim/proc/subjsets/fullset/train')
    datadir = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned'

    img_files = [os.path.join(datadir, subj, norm_filename) for subj in subjs]
    base_gen = vxm.generators.scan_to_scan(img_files, batch_size=1, add_feat_axis=True)

    while True:
        invols, outvols = next(base_gen)
        invols  = [prep(vol) for vol in invols]
        outvols = [prep(vol) for vol in outvols]
        yield (invols, outvols)


vol_shape = (160, 192, 224)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

device = configure_device(sys.argv[1])
with tf.device(device):

    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    losses = ['mse', vxm.losses.Grad('l2').loss]
    loss_weights = [1, 0.01]
    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
    vxm_model.fit_generator(train_generator(), epochs=10, steps_per_epoch=100, verbose=1)
    vxm_model.save_weights('3d_10.h5')
