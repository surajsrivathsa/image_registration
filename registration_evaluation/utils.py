import os, glob, sys
import numpy as np
import nibabel as nb
import re as re
import constants as const

class EvaluationUtils:
    def __init__(self, evaluation_folder, logging_flag = False, log = None):
        self.evaluation_folder = evaluation_folder
        self.regex_group_pattern = re.compile(const.REGEX_PATTERN)
        self.data_dict = {}
        self.dice_score_dict = {}
        self.fixed_file_list = None
        self.warped_file_list = None
        self.logging_flag = logging_flag
        self.log = log

    
    #{ixi001: {csf: [], wm: [], gm: []}, ixi002: {csf: [], gm: [], wm: []}, ixi003: {csf: [], gm: [], wm: []}}

    def calculateDiceScore(self):
        dice_score_dict = {}
        for filename, dict_val in self.data_dict.items():
            print()
            print("========= ============ ================")

            if(self.logging_flag):
                self.log.info("")
                self.log.info("========= ============ ================")

            self.dice_score_dict[filename] = {"dice_score_csf": 0.0, "dice_score_gm": 0.0, "dice_score_wm": 0.0, "dice_score_mean": 0.0}
            mean_dice_score = 0.0
            
            for segtype, images in dict_val.items():
                
                img1 = images[0]
                img2 = images[1]
                img1_intersect_img2 = np.sum(np.multiply(img1, img2))
                img1_union_img2 = np.sum(img1) + np.sum(img2)

                ##If denominator is also zero (in case all voxels are zero) then dice becomes infinity, avoid this
                if (img1_union_img2 == 0.0):
                    img1_union_img2 = 1.0
                
                dice_score = img1_intersect_img2 * 2.0 / img1_union_img2

                print("numerator: {}".format(img1_intersect_img2 * 2.0))
                print("denominator: {}".format( img1_union_img2 ))
                print('Dice similarity score for {} file with segment type {} is {}'.format(filename, segtype, dice_score))

                if(self.logging_flag):
                    self.log.info("numerator: {}".format(img1_intersect_img2 * 2.0))
                    self.log.info("denominator: {}".format( img1_union_img2 ))
                    self.log.info("Dice similarity score for {} file with segment type {} is {}".format(filename, segtype, dice_score))

                mean_dice_score = mean_dice_score + dice_score
                self.dice_score_dict[filename]["dice_score_" + segtype] = dice_score

            mean_dice_score = mean_dice_score / 3.0
            self.dice_score_dict[filename]["dice_score_mean"] = mean_dice_score
            print()
            print("Mean Dice score of {} is {}".format(filename, mean_dice_score))

            if(self.logging_flag):
                self.log.info("")
                self.log.info("Mean Dice score of {} is {}".format(filename, mean_dice_score))

        return;
 
    def processNiftifile(self, filepath):
        img_nb = nb.load(filepath)       
        img_np = img_nb.dataobj
        print("Shape and max intensity of img is {} and {}".format(img_np.shape, np.max(img_np)))

        #Do intensity normalization if intensities are > 1.0
        if(np.max(img_np) > 1.0):
            img_np = img_np/np.max(img_np)

        # print("Shape, max intensity and type of img is {}, {} and {}".format(img_np.shape, np.max(img_np), np.dtype(img_np)))
        
        if(self.logging_flag):
            self.log.info("")
            self.log.info("Shape and max intensity is {} and {}".format(img_np.shape, np.max(img_np)))
        
        return img_np

    def extractFilestoNumpyarrays(self):

        main_dict = {}
        for root, dirs, files in os.walk(self.evaluation_folder): 
            print()
            print("========= ============ =========")
            for fl in files:
                captured_groups = re.search(self.regex_group_pattern, fl)
                if captured_groups:
                    base_name = captured_groups.group(1)
                    contrast_type = captured_groups.group(2)
                    segmentation_type = captured_groups.group(3)
                    print(base_name + "|" + contrast_type + "|" +segmentation_type )
                    if(self.logging_flag):
                        self.log.info(base_name + "|" + contrast_type + "|" +segmentation_type )
                else:
                    #Skip if the file is not nifti or doesnt match regex pattern
                    continue;

                if base_name not in main_dict:
                    main_dict[base_name] = {"csf": [], "wm": [], "gm": []}
                
                img_np = self.processNiftifile(os.path.join(self.evaluation_folder, base_name, fl))
                if segmentation_type == "csf":
                    main_dict[base_name]["csf"].append(img_np) 
                elif segmentation_type == "wm":
                    main_dict[base_name]["wm"].append(img_np)
                elif segmentation_type == "gm":
                    main_dict[base_name]["gm"].append(img_np)
                else:
                    main_dict[base_name]["gm"].append(img_np)

            print()
        self.data_dict = main_dict
        print()
        print("========= ============ ========")
        print("Found unique eval folders: {}".format(self.data_dict.keys()))
        
        if(self.logging_flag):
            self.log.info("")
            self.log.info("========= ============ ========")
            self.log.info("Found unique eval folders: {}".format(self.data_dict.keys()))

        return ;







    
     