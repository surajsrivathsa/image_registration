import os, sys
import time
import numpy as np
import tensorflow as tf
import logging
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
print("Tensorflow version: {}".format(tf.__version__))

import neurite as ne
import nibabel as nb
import argparse
import nibabel.processing as nbp
import dl_voxelmorph_constants as const
sys.path.insert(0, const.FILEPATH_VOXELMORPH_LIB)
import voxelmorph as vxm
import img_processing as ipr
import img_registration as ireg

#vol_shape = (224, 320, 320)


class Driver:
    def __init__(self, stationary_img_path, moving_img_path, output_warped_image_path,
                loss_fnc, resampling_dim, logging_flag, padding_flag, reorient_flag, 
                resample_flag, resampling_type, model_path,
                gpu_flag, warp_file_path, multichannel):
        self.stationary_img_path = stationary_img_path
        self.moving_img_path = moving_img_path
        self.output_warped_image_path = output_warped_image_path
        self.loss_fnc = loss_fnc
        self.resampling_size = resampling_size
        self.logging_flag = logging_flag
        self.padding_flag = padding_flag
        self.reorient_flag = reorient_flag
        self.resample_flag = resample_flag
        self.resampling_type = resampling_type
        self.model_path = model_path
        self.warp_file_path = warp_file_path
        self.multichannel = multichannel
        self.device = None
        self.gpu_flag = gpu_flag
        self.nb_features = [ [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16] ]
        self.deformation_level = 1
        if(self.logging_flag):
            logging.basicConfig(filename= const.FILE_PATH_LOG, level = const.LOGGING_LEVEL, filemode=const.FILE_MODE, format='%(asctime)s - s%(name)s - %(levelname)s - %(message)s')
            self.log = logging.getLogger()
            self.log.info("Logging has been enabled")
            print("Logging has been enabled")
            print()
        else:
            print("Logging has been disabled")
            self.log = None
            print()

    def start_process(self):

        stationary_image_file_path = self.stationary_img_path
        moving_image_file_path = self.moving_img_path
        output_warped_image_file_path = self.output_warped_image_path
        warp_file_path = self.warp_file_path

        print()
        print(" ============= preprocessing started ===================")
        img_prcs_obj = ipr.ImageprocessingUtils(stationary_image_file_path=self.stationary_img_path, moving_image_file_path=self.moving_img_path, \
        output_warped_image_file_path = self.output_warped_image_path, reorient_flag=self.reorient_flag, resample_flag=self.resample_flag, \
        resampling_type=self.resampling_type,  resampling_size=self.resampling_size, padding_flag=self.padding_flag, logging_flag=self.logging_flag)

        orig_nii_stationary, orig_nii_moving = img_prcs_obj.read_input_images();
        canonical_img_1, canonical_img_2 = img_prcs_obj.reorient_images();
        resampled_stationary_img, resampled_moving_img = img_prcs_obj.resample_image();
        preprocessed_stationary_img, preprocessed_moving_img, img_shape, interim_img_1_path, interim_img_2_path = img_prcs_obj.pad_nifti_img()

        print()
        print(" ============= preprocessing completed ===================")
        print()
        print(" ============= starting registration ===================")
        img_regs_obj = ireg.ImageRegistrationUtils(interim_img_1_path, interim_img_2_path, self.model_path, self.gpu_flag, 
                                                   self.output_warped_image_path, self.warp_file_path, self.multichannel, img_shape, 
                                                   self.nb_features, self.deformation_level, self.logging_flag)
        img_regs_obj.register_and_save_warped_image();
        print(" ============= registration ended===================")
        print("")
        print(" ============= warped image saved to path ===================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser and parse the arguments from command line
    ap.add_argument( "--stationary_img_path", nargs= "?", required=False, help=" stationary_img_path", default = const.FILE_PATH_STATIONARY_IMG)
    ap.add_argument("--moving_img_path", nargs= "?", required=False, help="moving_img_path", default = const.FILE_PATH_MOVING_IMG)
    ap.add_argument("--output_warped_image_path", nargs= "?", required=False, help=" output_warped_image_path", default=const.FILE_PATH_WARPED_IMG)
    ap.add_argument("--loss_fnc", nargs= "?", required=False, help="loss_fnc", default = const.LOSS_FNC)
    ap.add_argument("--resampling_size", nargs= "?", required=False, help="resampling_dim", default = const.RESAMPLING_SIZE)
    ap.add_argument("--logging_flag", nargs= "?", required=False, help="logging_flag", default = "False")
    ap.add_argument("--padding_flag", nargs= "?", required=False, help="padding_flag", default = "True")
    ap.add_argument("--reorient_flag", nargs= "?", required=False, help="reorient_flag", default = "True")
    ap.add_argument("--resample_flag", nargs= "?", required=False, help="resample_flag", default = "True")
    ap.add_argument("--resampling_type", nargs= "?", required=False, help="resampling_type", default = const.RESAMPLING_TYPE)
    ap.add_argument("--model_path", nargs= "?", required=False, help="model_path", default = const.MODEL_PATH)
    ap.add_argument("--warp_file_path", nargs= "?", required=False, help="deformation_field_file", default = const.WARP_FILE_PATH)
    ap.add_argument("--gpu", nargs= "?", required=False, help="gpu", default = const.GPU)
    ap.add_argument("--multichannel", nargs= "?", required=False, help="multichannel", default = const.MULTICHANNEL)
    args = vars(ap.parse_args())

    stationary_img_path = args["stationary_img_path"]
    moving_img_path = args["moving_img_path"]
    output_warped_image_path = args["output_warped_image_path"]
    loss_fnc = args["loss_fnc"]
    resampling_size = args["resampling_size"]
    resampling_type = args["resampling_type"]
    model_path = args['model_path']
    warp_file_path = args['warp_file_path']
    gpu = args['gpu']
    multichannel = args['multichannel']


    padding_flag = args["padding_flag"]
    if(padding_flag == "True"):
        padding_flag = True
    else:
        padding_flag = False

    reorient_flag = args["reorient_flag"]
    if(reorient_flag == "True"):
        reorient_flag = True
    else:
        reorient_flag = False

    resample_flag = args["resample_flag"]
    if(resample_flag == "True"):
        resample_flag = True
    else:
        resample_flag = False

    logging_flag = args["logging_flag"]
    if(logging_flag == "True"):
        logging_flag = True
    else:
        logging_flag = False
    
    start_time = time.time()

    driver_obj = Driver(stationary_img_path, moving_img_path, output_warped_image_path, loss_fnc, resampling_size, 
                        logging_flag, padding_flag, reorient_flag, resample_flag, resampling_type,
                        model_path=model_path,warp_file_path=warp_file_path,gpu_flag=gpu, multichannel=multichannel)
    driver_obj.start_process()
    print("--- %s seconds ---" % (time.time() - start_time))