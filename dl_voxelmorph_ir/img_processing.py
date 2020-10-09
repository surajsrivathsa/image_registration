import os
import sys
import logging
import nibabel as nb
import numpy as np
import tensorflow as tf
import math
print(tf.__version__)
import dl_voxelmorph_constants as const
sys.path.insert(0, const.FILEPATH_VOXELMORPH_LIB)
import voxelmorph as vxm

class ImageprocessingUtils:
    
    def __init__(self, stationary_image_file_path, moving_image_file_path,
    output_warped_image_file_path, reorient_flag, resample_flag, resampling_type,  resampling_size,
    padding_flag, logging_flag, log = None):

        self.stationary_image_file_path = stationary_image_file_path
        self.moving_image_file_path = moving_image_file_path
        self.output_warped_image_file_path = output_warped_image_file_path
        self.reorient_flag = reorient_flag
        self.resample_flag = resample_flag
        self.resampling_type = resampling_type
        self.logging_flag = logging_flag
        self.log = log
        self.orig_nii_stationary = None
        self.orig_nii_moving = None
        self.canonical_img_1 = None
        self.canonical_img_2 = None
        self.resampled_stationary_img = None
        self.resampled_moving_img = None
        self.resampling_size=resampling_size
        self.padding_flag = padding_flag
        self.padded_stationary_img_np = None
        self.padded_moving_img_np = None
        self.padded_stationary_img = None
        self.padded_moving_img = None
        self.stationary_img_voxel_dim = None
        self.moving_img_voxel_dim = None
        self.stationary_img_centre = None
        self.moving_img_centre = None
        self.img_shape = None
        self.affine_transformation_matrix = None
        self.affine_transformation_object = None
        self.vol_shape = (224, 320, 320)
        self.nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]

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

    def read_input_images(self):
        orig_nii_stationary = nb.load(self.stationary_image_file_path)
        orig_nii_moving = nb.load(self.moving_image_file_path)

        orig_nii_stationary_voxel_dim = orig_nii_stationary.header["pixdim"][1:4]
        orig_nii_moving_voxel_dim = orig_nii_moving.header["pixdim"][1:4]
        orig_nii_stationary_centre = [float(orig_nii_stationary.header["qoffset_x"]), float(orig_nii_stationary.header["qoffset_y"]), float(orig_nii_stationary.header["qoffset_z"])]
        orig_nii_moving_centre = [float(orig_nii_moving.header["qoffset_x"]), float(orig_nii_moving.header["qoffset_y"]), float(orig_nii_moving.header["qoffset_z"])]


        print(" ============= ============== ===================")
        print("Image 1 voxel resolution before resampling: {}".format(orig_nii_stationary_voxel_dim))
        print(" ============= ============== ===================")
        print("Image 2 voxel resolution before resampling: {}".format(orig_nii_moving_voxel_dim))
        print(" ============= ============== ===================")
        print("Image 1 centre before resampling: {}".format(orig_nii_stationary_centre))
        print(" ============= ============== ===================")
        print("Image 2 centre before resampling: {}".format(orig_nii_moving_centre))   
        print(" ============= ============== ===================")
        print("original t1 affine: {}".format(orig_nii_stationary.affine))
        print(" ============= ============== ===================")
        print("original t2 affine: {}".format(orig_nii_moving.affine))
        print(" ============= ============== ===================")
        print("original t1 Orientation: {}".format(nb.aff2axcodes(orig_nii_stationary.affine)))
        print(" ============= ============== ===================")
        print("original t2 Orientation: {}".format(nb.aff2axcodes(orig_nii_moving.affine)))
        print(" ============= ============== ===================")

        self.orig_nii_stationary = orig_nii_stationary
        self.orig_nii_moving = orig_nii_moving

        return self.orig_nii_stationary, self.orig_nii_moving;


    def reorient_images(self):
        if(self.reorient_flag):
            print("Reorient flag is set to true, Hence reorienting both images to Right Anterior Superior")
            canonical_img_1 = nb.as_closest_canonical(self.orig_nii_stationary)
            print(" ============= ============== ===================")
            print("orientation changed  t1 affine: {}".format(canonical_img_1.affine))
            print(" ============= ============== ===================")
            print("orientation changed  t1 : {}".format(nb.aff2axcodes(canonical_img_1.affine)))
            print(" ============= ============== ===================")
            canonical_img_2 = nb.as_closest_canonical(self.orig_nii_moving)
            print(" ============= ============== ===================")
            print("orientation changed  t2 affine: {}".format(canonical_img_2.affine))
            print(" ============= ============== ===================")
            print("orientation changed  t1 : {}".format(nb.aff2axcodes(canonical_img_2.affine)))
            print(" ============= ============== ===================")

            self.canonical_img_1 = canonical_img_1
            self.canonical_img_2 = canonical_img_2
            return self.canonical_img_1, self.canonical_img_2
        else:
            print(" ============= ============== ===================")
            print("Not reorienting the images as reorient flag is false")
            print(" ============= ============== ===================")
            self.canonical_img_1 = orig_nii_stationary
            self.canonical_img_2 = orig_nii_moving
            return self.canonical_img_1, self.canonical_img_2;


    def resample_image(self):
        if(self.resample_flag):
            if(self.resampling_type == "both_moving_and_stationary"):
                print("Chosen resampling method: {}".format(self.resampling_type))
                resampled_voxel_size = self.resampling_size
                canonical_img_1 = nb.processing.resample_to_output(self.canonical_img_1,voxel_sizes=resampled_voxel_size)
                canonical_img_2 = nb.processing.resample_to_output(self.canonical_img_2,voxel_sizes=resampled_voxel_size)

                print(" ============= ============== ===================")
                print("Shape of resampled 1 image: {}".format(canonical_img_1.header.get_data_shape()))
                print(" ============= ============== ===================")
                print("resampled t1 affine: {}".format(canonical_img_1.affine))
                print(" ============= ============== ===================")
                print("Shape of resampled 1 image: {}".format(canonical_img_2.header.get_data_shape()))
                print(" ============= ============== ===================")       
                print("resampled 2 affine: {}".format(canonical_img_2.affine))
                print(" ============= ============== ===================")

                ci1_shape = canonical_img_1.header.get_data_shape()
                ci2_shape = canonical_img_2.header.get_data_shape()
                max_shapes = (max(ci1_shape[0], ci2_shape[0]), max(ci1_shape[1], ci2_shape[1]), max(ci1_shape[2], ci2_shape[2]))
                max_shapes_array = [max(ci1_shape[0], ci2_shape[0]), max(ci1_shape[1], ci2_shape[1]), max(ci1_shape[2], ci2_shape[2])]
                
                canonical_img_1_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_2_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_1_centre = [float(canonical_img_1.header["qoffset_x"]), float(canonical_img_1.header["qoffset_y"]), float(canonical_img_1.header["qoffset_z"])]
                canonical_img_2_centre = [float(canonical_img_2.header["qoffset_x"]), float(canonical_img_2.header["qoffset_y"]), float(canonical_img_2.header["qoffset_z"])]

                print(" ============= ============== ===================")
                print("Image 1 voxel resolution after resampling: {}".format(canonical_img_1_voxel_dim))
                print(" ============= ============== ===================")
                print("Image 2 voxel resolution after resampling: {}".format(canonical_img_2_voxel_dim))
                print(" ============= ============== ===================")
                print("Image 1 centre after resampling: {}".format(canonical_img_1_centre))
                print(" ============= ============== ===================")
                print("Image 2 centre after resampling: {}".format(canonical_img_2_centre))

                self.resampled_stationary_img = canonical_img_1
                self.resampled_moving_img = canonical_img_2
                self.stationary_img_voxel_dim = canonical_img_1_voxel_dim
                self.moving_img_voxel_dim = canonical_img_2_voxel_dim
                self.stationary_img_centre = canonical_img_1_centre
                self.moving_img_centre = canonical_img_2_centre

                return self.resampled_stationary_img, self.resampled_moving_img;
                
            elif(self.resampling_type == "only_moving"):
                print("Chosen resampling method: {}".format(self.resampling_type))
                resampled_voxel_size = self.resampling_size
                canonical_img_1 = self.canonical_img_1
                canonical_img_2 = nb.processing.resample_to_output(self.canonical_img_2,voxel_sizes=resampled_voxel_size)
                print(" ============= ============== ===================")
                print("Shape of resampled 1 image: {}".format(canonical_img_1.header.get_data_shape()))
                print(" ============= ============== ===================")
                print("resampled t1 affine: {}".format(canonical_img_1.affine))
                print(" ============= ============== ===================")
                print("Shape of resampled 1 image: {}".format(canonical_img_2.header.get_data_shape()))
                print(" ============= ============== ===================")       
                print("resampled 2 affine: {}".format(canonical_img_2.affine))
                print(" ============= ============== ===================")

                ci1_shape = canonical_img_1.header.get_data_shape()
                ci2_shape = canonical_img_2.header.get_data_shape()
                max_shapes = (max(ci1_shape[0], ci2_shape[0]), max(ci1_shape[1], ci2_shape[1]), max(ci1_shape[2], ci2_shape[2]))
                max_shapes_array = [max(ci1_shape[0], ci2_shape[0]), max(ci1_shape[1], ci2_shape[1]), max(ci1_shape[2], ci2_shape[2])]
                
                canonical_img_1_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_2_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_1_centre = [float(canonical_img_1.header["qoffset_x"]), float(canonical_img_1.header["qoffset_y"]), float(canonical_img_1.header["qoffset_z"])]
                canonical_img_2_centre = [float(canonical_img_2.header["qoffset_x"]), float(canonical_img_2.header["qoffset_y"]), float(canonical_img_2.header["qoffset_z"])]

                print(" ============= ============== ===================")
                print("Image 1 voxel resolution after resampling: {}".format(canonical_img_1_voxel_dim))
                print(" ============= ============== ===================")
                print("Image 2 voxel resolution after resampling: {}".format(canonical_img_2_voxel_dim))
                print(" ============= ============== ===================")
                print("Image 1 centre after resampling: {}".format(canonical_img_1_centre))
                print(" ============= ============== ===================")
                print("Image 2 centre after resampling: {}".format(canonical_img_2_centre))

                self.resampled_stationary_img = canonical_img_1
                self.resampled_moving_img = canonical_img_2
                self.stationary_img_voxel_dim = canonical_img_1_voxel_dim
                self.moving_img_voxel_dim = canonical_img_2_voxel_dim
                self.stationary_img_centre = canonical_img_1_centre
                self.moving_img_centre = canonical_img_2_centre
                return self.resampled_stationary_img, self.resampled_moving_img;

            elif(self.resampling_type == "match_to_stationary"):
                print("Chosen resampling method: {}".format(self.resampling_type))
                canonical_img_1 = self.canonical_img_1
                canonical_img_2 = nilearn.image.resample_to_img(self.canonical_img_2, self.canonical_img_1)
                print(" ============= ============== ===================")
                print("Shape of resampled 1 image: {}".format(canonical_img_1.header.get_data_shape()))
                print(" ============= ============== ===================")
                print("resampled t1 affine: {}".format(canonical_img_1.affine))
                print(" ============= ============== ===================")
                print("Shape of resampled 1 image: {}".format(canonical_img_2.header.get_data_shape()))
                print(" ============= ============== ===================")       
                print("resampled 2 affine: {}".format(canonical_img_2.affine))
                print(" ============= ============== ===================")

                ci1_shape = canonical_img_1.header.get_data_shape()
                ci2_shape = canonical_img_2.header.get_data_shape()
                max_shapes = (max(ci1_shape[0], ci2_shape[0]), max(ci1_shape[1], ci2_shape[1]), max(ci1_shape[2], ci2_shape[2]))
                max_shapes_array = [max(ci1_shape[0], ci2_shape[0]), max(ci1_shape[1], ci2_shape[1]), max(ci1_shape[2], ci2_shape[2])]
                
                canonical_img_1_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_2_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_1_centre = [float(canonical_img_1.header["qoffset_x"]), float(canonical_img_1.header["qoffset_y"]), float(canonical_img_1.header["qoffset_z"])]
                canonical_img_2_centre = [float(canonical_img_2.header["qoffset_x"]), float(canonical_img_2.header["qoffset_y"]), float(canonical_img_2.header["qoffset_z"])]

                print(" ============= ============== ===================")
                print("Image 1 voxel resolution after resampling: {}".format(canonical_img_1_voxel_dim))
                print(" ============= ============== ===================")
                print("Image 2 voxel resolution after resampling: {}".format(canonical_img_2_voxel_dim))
                print(" ============= ============== ===================")
                print("Image 1 centre after resampling: {}".format(canonical_img_1_centre))
                print(" ============= ============== ===================")
                print("Image 2 centre after resampling: {}".format(canonical_img_2_centre))

                self.resampled_stationary_img = canonical_img_1
                self.resampled_moving_img = canonical_img_2
                self.stationary_img_voxel_dim = canonical_img_1_voxel_dim
                self.moving_img_voxel_dim = canonical_img_2_voxel_dim
                self.stationary_img_centre = canonical_img_1_centre
                self.moving_img_centre = canonical_img_2_centre
                return self.resampled_stationary_img, self.resampled_moving_img;
                
            else:
                canonical_img_1_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_2_voxel_dim = canonical_img_1.header["pixdim"][1:4]
                canonical_img_1_centre = [float(canonical_img_1.header["qoffset_x"]), float(canonical_img_1.header["qoffset_y"]), float(canonical_img_1.header["qoffset_z"])]
                canonical_img_2_centre = [float(canonical_img_2.header["qoffset_x"]), float(canonical_img_2.header["qoffset_y"]), float(canonical_img_2.header["qoffset_z"])]

                print("Chosen resampling method: {}".format(self.resampling_type))
                print("Not a valid reorientation type, returning the older images")
                self.resampled_stationary_img = self.canonical_img_1
                self.resampled_moving_img = self.canonical_img_2

                self.stationary_img_voxel_dim = canonical_img_1_voxel_dim
                self.moving_img_voxel_dim = canonical_img_2_voxel_dim
                self.stationary_img_centre = canonical_img_1_centre
                self.moving_img_centre = canonical_img_2_centre
                return self.resampled_stationary_img, self.resampled_moving_img;
        
        else:
            canonical_img_1_voxel_dim = canonical_img_1.header["pixdim"][1:4]
            canonical_img_2_voxel_dim = canonical_img_1.header["pixdim"][1:4]
            canonical_img_1_centre = [float(canonical_img_1.header["qoffset_x"]), float(canonical_img_1.header["qoffset_y"]), float(canonical_img_1.header["qoffset_z"])]
            canonical_img_2_centre = [float(canonical_img_2.header["qoffset_x"]), float(canonical_img_2.header["qoffset_y"]), float(canonical_img_2.header["qoffset_z"])]

            print("Returning older images as you have choosen not to resample")
            self.resampled_stationary_img = self.canonical_img_1
            self.resampled_moving_img = self.canonical_img_2

            self.stationary_img_voxel_dim = canonical_img_1_voxel_dim
            self.moving_img_voxel_dim = canonical_img_2_voxel_dim
            self.stationary_img_centre = canonical_img_1_centre
            self.moving_img_centre = canonical_img_2_centre
            return self.resampled_stationary_img, self.resampled_moving_img;


    def find_correct_shape(self,input_shapes, nb_vol_max_dim=32):
        max_shapes = []
        max_shapes.append(math.ceil(input_shapes[0] * 1.0/nb_vol_max_dim) * nb_vol_max_dim);
        max_shapes.append(math.ceil(input_shapes[1] * 1.0/nb_vol_max_dim) * nb_vol_max_dim);
        max_shapes.append(math.ceil(input_shapes[2] * 1.0/nb_vol_max_dim) * nb_vol_max_dim);
        print("Max shapes: {}".format(max_shapes))
        return tuple(max_shapes);


    def convert_tensor_to_nifti(self, warped_img_tnsr, transformation, displacement):
        print("=============== transformation matrix before ==============================")
        self.affine_transformation_matrix = transformation.transformation_matrix
        print(self.affine_transformation_matrix)
        warped_img_np = warped_img_tnsr.numpy();
        affine_transformation_matrix_np = transformation.transformation_matrix.detach().cpu().numpy()
        final_transformation_matrix = np.identity(4)
        final_transformation_matrix[0:3, :] = affine_transformation_matrix_np[: , :]

        print("============== transformation matrix after ==================")
        print(final_transformation_matrix)

        warped_nifti_img = nb.Nifti1Image(warped_img_np, affine=final_transformation_matrix)
        print("============== ========================== ==================")

        return warped_nifti_img;


    def save_warped_image(self, warped_nifti_img):
        print("Saving file to : {}".format(self.output_warped_image_file_path))
        warped_nifti_img.to_filename(self.output_warped_image_file_path)
        return;


    def pad_nifti_img(self):
        if(self.padding_flag):
            print("padding all tensors to their nb vol dims per dimensions")
            stationary_img_shape = self.resampled_stationary_img.header.get_data_shape()
            moving_img_shape = self.resampled_moving_img.header.get_data_shape()
            max_shapes = self.find_correct_shape((max(stationary_img_shape[0], moving_img_shape[0]), max(stationary_img_shape[1], moving_img_shape[1]), max(stationary_img_shape[2], moving_img_shape[2])), nb_vol_max_dim=32)
            max_shapes_array = list(max_shapes)
            stationary_img_np = np.array(self.resampled_stationary_img.dataobj)
            moving_img_np = np.array(self.resampled_moving_img.dataobj)
            mask1_img_np = np.zeros(max_shapes)
            mask2_img_np = np.zeros(max_shapes)
            mask1_img_np[:stationary_img_shape[0], :stationary_img_shape[1], :stationary_img_shape[2]] = stationary_img_np
            mask2_img_np[:moving_img_shape[0], :moving_img_shape[1], :moving_img_shape[2]] = moving_img_np

            mask1_img_np = mask1_img_np/np.max(mask1_img_np)
            mask2_img_np = mask2_img_np/np.max(mask2_img_np)
            self.padded_stationary_img_np = mask1_img_np
            self.padded_moving_img_np = mask2_img_np
            self.img_shape =  max_shapes_array

            self.padded_stationary_img = nb.Nifti1Image(mask1_img_np, affine=self.resampled_stationary_img.affine)
            self.padded_moving_img = nb.Nifti1Image(mask2_img_np, affine=self.resampled_moving_img.affine)

            self.padded_stationary_img.to_filename(str(os.path.join(os.getcwd(), 'interim_img1.nii.gz')))
            self.padded_moving_img.to_filename(str(os.path.join(os.getcwd(), 'interim_img2.nii.gz')))

            interim_img_1_path = str(os.path.join(os.getcwd(), 'interim_img1.nii.gz'))
            interim_img_2_path = str(os.path.join(os.getcwd(), 'interim_img2.nii.gz'))

            return self.padded_stationary_img, self.padded_moving_img, self.img_shape, interim_img_1_path, interim_img_2_path;

        else:
            print(" ============= ============== ===================")
            print("Not padding as pad flag is false")
            stationary_img_shape = self.resampled_stationary_img.header.get_data_shape()
            moving_img_shape = self.resampled_moving_img.header.get_data_shape()
            max_shapes = self.find_correct_shape((max(stationary_img_shape[0], moving_img_shape[0]), max(stationary_img_shape[1], moving_img_shape[1]), max(stationary_img_shape[2], moving_img_shape[2])), nb_vol_max_dim=32)
            max_shapes_array = list(max_shapes)
            stationary_img_np = np.array(self.resampled_stationary_img.dataobj)
            moving_img_np = np.array(self.resampled_moving_img.dataobj)
            stationary_img_np = stationary_img_np/np.max(stationary_img_np)
            moving_img_np = moving_img_np/np.max(moving_img_np)
            self.padded_stationary_img_np = stationary_img_np
            self.padded_moving_img_np = moving_img_np
            self.img_shape =  max_shapes_array

            self.padded_stationary_img = nb.Nifti1Image(self.padded_stationary_img_np, affine=self.resampled_stationary_img.affine)
            self.padded_moving_img = nb.Nifti1Image(self.padded_moving_img_np, affine=self.resampled_moving_img.affine)

            self.padded_stationary_img.to_filename(str(os.path.join(os.getcwd(), 'interim_img1.nii.gz')))
            self.padded_moving_img.to_filename(str(os.path.join(os.getcwd(), 'interim_img2.nii.gz')))

            interim_img_1_path = str(os.path.join(os.getcwd(), 'interim_img1.nii.gz'))
            interim_img_2_path = str(os.path.join(os.getcwd(), 'interim_img2.nii.gz'))
            return self.padded_stationary_img, self.padded_moving_img, self.img_shape, interim_img_1_path, interim_img_2_path;
