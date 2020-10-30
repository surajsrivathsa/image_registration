import os, glob
import numpy as np
import nibabel as nb
print("nibabel version: {}".format(nb.__version__))
import utils 
import driver
import constants as const
import logging
 

if __name__ == "__main__":
    evaluation_folder = const.FOLDERPATH_EVALUATION
    logging_flag = const.LOGGING_FLAG
    eval_driver_obj = driver.EvaluationDriver(evaluation_folder = evaluation_folder, logging_flag=logging_flag)
    eval_driver_obj.runEvaluation()