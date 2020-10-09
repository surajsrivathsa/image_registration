import os, sys
import numpy as np
import tensorflow as tf
import logging
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
print("Tensorflow version: {}".format(tf.__version__))

import neurite as ne
import nibabel as nb
import nibabel.processing as nbp
import dl_voxelmorph_constants as const
sys.path.insert(0, const.FILEPATH_VOXELMORPH_LIB)
import voxelmorph as vxm



class ImageRegistrationUtils:
    def __init__(self, preprocessed_stationary_img, preprocessed_moving_img, model_path,
                 gpu_flag, moved_file_path, warp_file_path, multichannel, img_shape, nb_features,
                deformation_level, logging_flag=const.LOGGING_FLAG, log = None):
        self.preprocessed_stationary_img = preprocessed_stationary_img
        self.preprocessed_moving_img = preprocessed_moving_img
        self.model_path = model_path
        self.gpu_flag = gpu_flag
        self.device = None
        self.deformation_field_file_path = warp_file_path
        self.warped_image_file_path = moved_file_path
        self.multichannel = multichannel
        self.img_shape = img_shape
        self.nb_feats = nb_features
        self.deformation_level = deformation_level
        self.logging_flag = logging_flag
        self.log = log
        


    def register_and_save_warped_image(self):
        # tensorflow device handling
        self.device, nb_devices = vxm.tf.utils.setup_device(self.gpu_flag)

        print(" ============= ==================== ===================")
        print("device: {}, nb_devices: {}".format(self.device, nb_devices))
        print(" ============= ==================== ===================")
        print("chosen model: {}".format(self.model_path))
        print("")
        # load moving and fixed images
        add_feat_axis = not self.multichannel
        moving, moving_affine = vxm.py.utils.load_volfile(self.preprocessed_moving_img, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        fixed, fixed_affine = vxm.py.utils.load_volfile(self.preprocessed_stationary_img, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

        inshape = moving.shape[1:-1]
        nb_feats = moving.shape[-1]

        with tf.device(self.device):
            # load model and predict
            vxm_model = vxm.networks.VxmDense(self.img_shape, self.nb_feats, int_steps=self.deformation_level);
            vxm_model.load_weights(self.model_path)
            print(vxm_model.summary())
            warp = vxm_model.register(moving, fixed)
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

        # save warp if path is provided
        if self.deformation_field_file_path:
            vxm.py.utils.save_volfile(warp.squeeze(), self.deformation_field_file_path, fixed_affine)

        # save moved image
        vxm.py.utils.save_volfile(moved.squeeze(), self.warped_image_file_path, fixed_affine)