import torch
import sys
import os
import time
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nb
import argparse
import numpy as np
import nibabel.processing as nbp
import img_processing as ipr
import img_registration as ireg
import dl_airlabs_constants as const
sys.path.insert(0, const.FILEPATH_AIRLABS_LIB)
import airlab as al

class Driver:
    def __init__(self, stationary_img_path, moving_img_path, output_warped_image_path,
                loss_fnc, resampling_dim, logging_flag, padding_flag, reorient_flag, 
                resample_flag, resampling_type):
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

        stationary_image_file_path = self.stationary_img_path
        moving_image_file_path = self.moving_img_path
        output_warped_image_file_path = self.output_warped_image_path

        print()
        print(" ============= preprocessing started ===================")
        img_prcs_obj = ipr.ImageprocessingUtils(stationary_image_file_path=self.stationary_img_path, moving_image_file_path=self.moving_img_path, \
        output_warped_image_file_path = self.output_warped_image_path, reorient_flag=self.reorient_flag, resample_flag=self.resample_flag, \
        resampling_type=self.resampling_type,  resampling_size=self.resampling_size, padding_flag=self.padding_flag, logging_flag=self.logging_flag)

        orig_nii_stationary, orig_nii_moving = img_prcs_obj.read_input_images();
        canonical_img_1, canonical_img_2 = img_prcs_obj.reorient_images();
        resampled_stationary_img, resampled_moving_img = img_prcs_obj.resample_image();
        preprocessed_stationary_img_tnsr, preprocessed_moving_img_tnsr, preprocessed_stationary_img_voxel_dim, preprocessed_moving_img_voxel_dim, preprocessed_stationary_img_centre, preprocessed_moving_img_centre, img_shape = img_prcs_obj.convert_nifti_to_tensor();

        print()
        print(" ============= preprocessing completed ===================")
        print()
        print(" ============= starting registration ===================")
        img_regs_obj = ireg.ImageRegistrationUtils(preprocessed_stationary_img_tnsr, preprocessed_moving_img_tnsr, preprocessed_stationary_img_voxel_dim, preprocessed_moving_img_voxel_dim, preprocessed_stationary_img_centre, preprocessed_moving_img_centre, img_shape, device, loss_fnc)
        warped_img_tnsr, transformation, displacement = img_regs_obj.three_dim_affine_reg();
        print(" ============= registration ended===================")
        print()
        print(" ============= starting post processing and saving warped image to disk ===================")
        print()
        warped_nifti_img = img_prcs_obj.convert_tensor_to_nifti(warped_img_tnsr, transformation, displacement)
        img_prcs_obj.save_warped_image(warped_nifti_img)
        print()
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
    
    args = vars(ap.parse_args())

    stationary_img_path = args["stationary_img_path"]
    moving_img_path = args["moving_img_path"]
    output_warped_image_path = args["output_warped_image_path"]
    loss_fnc = args["loss_fnc"]
    resampling_size = args["resampling_size"]
    resampling_type = args["resampling_type"]

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

    driver_obj = Driver(stationary_img_path, moving_img_path, output_warped_image_path, loss_fnc, resampling_size, logging_flag, padding_flag, reorient_flag, resample_flag, resampling_type)
    driver_obj.start_process()
    print("--- %s seconds ---" % (time.time() - start_time))
