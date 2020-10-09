import logging
import os
from pickle import FALSE
import sys

# File path and name constants
FILE_PATH_STATIONARY_IMG = "/Users/surajshashidhar/git/image_registration/T1_images/IXI002-Guys-0828-T1.nii.gz"
FILE_PATH_MOVING_IMG = "/Users/surajshashidhar/git/image_registration/T2_images/IXI002-Guys-0828-T2.nii.gz"
FILE_PATH_WARPED_IMG = "/Users/surajshashidhar/git/image_registration/warped_image_voxelmorph.nii.gz"
FILE_PATH_LOG = os.path.join(os.getcwd(), 'AIRLABS_Image_Registration_log.log')
FILE_MODE = "w"
FILEPATH_VOXELMORPH_LIB = '/Users/surajshashidhar/git/voxelmorph'


LOGGING_LEVEL = 20
RESAMPLING_SIZE = [1, 1, 1]
LOSS_FNC = "MSE"
REORIENT_FLAG=True
RESAMPLE_FLAG=True
RESAMPLING_TYPE="both_moving_and_stationary"
PADDING_FLAG=True
LOGGING_FLAG=False
MODEL_PATH = "/Users/surajshashidhar/git/image_registration/brain_3d.h5"
WARP_FILE_PATH = "/Users/surajshashidhar/git/image_registration/deformation_field.nii.gz"
GPU = None
MULTICHANNEL = False