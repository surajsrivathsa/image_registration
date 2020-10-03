import logging
import os
import sys
import time

# File path and name constants
FILE_PATH_STATIONARY_IMG = "/Users/surajshashidhar/git/image_registration/T1_images/IXI002-Guys-0828-T1.nii.gz"
FILE_PATH_MOVING_IMG = "/Users/surajshashidhar/git/image_registration/T2_images/IXI002-Guys-0828-T2.nii.gz"
FILE_PATH_WARPED_IMG = "/Users/surajshashidhar/git/image_registration/warped_image.nii.gz"
FILE_PATH_LOG = os.path.join(os.getcwd(), 'AIRLABS_Image_Registration_log.log')
FILE_MODE = "w"
FILEPATH_AIRLABS_LIB = '/Users/surajshashidhar/git/airlab'


LOGGING_LEVEL = 20
RESAMPLING_SIZE = [1, 1, 1]
LOSS_FNC = "MSE"
REORIENT_FLAG=True
RESAMPLE_FLAG=True
RESAMPLING_TYPE="both_moving_and_stationary"
PADDING_FLAG=True
LOGGING_FLAG=False
