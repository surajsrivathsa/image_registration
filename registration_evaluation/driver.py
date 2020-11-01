import os, glob
import numpy as np
import nibabel as nb
print("nibabel version: {}".format(nb.__version__))
import utils 
import constants as const
import logging


class EvaluationDriver:
    def __init__(self, evaluation_folder, fixed_images_name_remove_str = "-T1_bet_resampled.nii.gz", 
    warped_images_name_remove_str = "T2_bet_resampled_registered.nii.gz", resampling_folder = None, triplet_folder = const.FOLDERPATH_TRIPLET, logging_flag = False):
        self.evaluation_folder = evaluation_folder
        self.dice_score_dict = {}
        self.fixed_images_name_remove_str = fixed_images_name_remove_str
        self.warped_images_name_remove_str = warped_images_name_remove_str
        self.resampling_folder = resampling_folder
        self.logging_flag = logging_flag
        self.triplet_folder = triplet_folder

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


    def runEvaluation(self):
        utils_obj = utils.EvaluationUtils(self.evaluation_folder, self.resampling_folder, self.triplet_folder, self.logging_flag, self.log)
        #utils_obj.extractFilestoNumpyarrays()
        #utils_obj.calculateDiceScore()
        #utils_obj.resampleNiftifile()
        utils_obj.diceScoreforTriplets()
        return;


"""
if __name__ == "__main__":
    evaluation_folder = const.FOLDERPATH_EVALUATION
    logging_flag = const.LOGGING_FLAG
    eval_driver_obj = EvaluationDriver(evaluation_folder = evaluation_folder, logging_flag=logging_flag)
    eval_driver_obj.runEvaluation()


"""