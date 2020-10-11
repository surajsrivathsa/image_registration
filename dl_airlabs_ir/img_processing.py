import os
import sys
import nibabel as nb
import numpy as np
import torch
import dl_airlabs_constants as const
sys.path.insert(0, const.FILEPATH_AIRLABS_LIB)
import airlab as al


#"/Users/surajshashidhar/git/image_registration/warped_image.nii.gz"

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
        self.padded_stationary_img_tnsr = None
        self.padded_moving_img_tnsr = None
        self.stationary_img_voxel_dim = None
        self.moving_img_voxel_dim = None
        self.stationary_img_centre = None
        self.moving_img_centre = None
        self.img_shape = None
        self.affine_transformation_matrix = None
        self.affine_transformation_object = None

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
            self.canonical_img_1 = self.orig_nii_stationary
            self.canonical_img_2 = self.orig_nii_moving
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
            canonical_img_1_voxel_dim = self.canonical_img_1.header["pixdim"][1:4]
            canonical_img_2_voxel_dim = self.canonical_img_1.header["pixdim"][1:4]
            canonical_img_1_centre = [float(self.canonical_img_1.header["qoffset_x"]), float(self.canonical_img_1.header["qoffset_y"]), float(self.canonical_img_1.header["qoffset_z"])]
            canonical_img_2_centre = [float(self.canonical_img_2.header["qoffset_x"]), float(self.canonical_img_2.header["qoffset_y"]), float(self.canonical_img_2.header["qoffset_z"])]

            print("Returning older images as you have choosen not to resample")
            self.resampled_stationary_img = self.canonical_img_1
            self.resampled_moving_img = self.canonical_img_2

            self.stationary_img_voxel_dim = canonical_img_1_voxel_dim
            self.moving_img_voxel_dim = canonical_img_2_voxel_dim
            self.stationary_img_centre = canonical_img_1_centre
            self.moving_img_centre = canonical_img_2_centre
            return self.resampled_stationary_img, self.resampled_moving_img;


    def convert_nifti_to_tensor(self):
        if(self.padding_flag):
            print("padding all tensors to their max values per dimensions")
            stationary_img_shape = self.resampled_stationary_img.header.get_data_shape()
            moving_img_shape = self.resampled_moving_img.header.get_data_shape()
            max_shapes = (max(stationary_img_shape[0], moving_img_shape[0]), max(stationary_img_shape[1], moving_img_shape[1]), max(stationary_img_shape[2], moving_img_shape[2]))
            max_shapes_array = [max(stationary_img_shape[0], moving_img_shape[0]), max(stationary_img_shape[1], moving_img_shape[1]), max(stationary_img_shape[2], moving_img_shape[2])]
      
            stationary_img_np = np.array(self.resampled_stationary_img.dataobj)
            moving_img_np = np.array(self.resampled_moving_img.dataobj)

            stationary_img_np = stationary_img_np * 1.0/np.max(stationary_img_np)
            moving_img_np = moving_img_np * 1.0/np.max(moving_img_np)

            stationary_img_tnsr = torch.from_numpy(stationary_img_np)        
            moving_img_tnsr = torch.from_numpy(moving_img_np)  

            padded_stationary_img_tnsr = torch.zeros(max_shapes[0], max_shapes[1], max_shapes[2])
            padded_moving_img_tnsr = torch.zeros(max_shapes[0], max_shapes[1], max_shapes[2])

            padded_stationary_img_tnsr[:stationary_img_tnsr.shape[0], :stationary_img_tnsr.shape[1], :stationary_img_tnsr.shape[2]] = stationary_img_tnsr
            padded_moving_img_tnsr[:moving_img_tnsr.shape[0], :moving_img_tnsr.shape[1], :moving_img_tnsr.shape[2]] = moving_img_tnsr

            padded_stationary_img_tnsr = padded_stationary_img_tnsr.type(torch.FloatTensor)
            padded_moving_img_tnsr = padded_moving_img_tnsr.type(torch.FloatTensor)
            print(" ============= ============== ===================")
            print(padded_stationary_img_tnsr.shape)
            print(" ============= ============== ===================")
            print(padded_moving_img_tnsr.shape)
            print(" ============= ============== ===================")

            print(padded_stationary_img_tnsr.type())
            print(" ============= ============== ===================")
            print(padded_moving_img_tnsr.type())
            print(" ============= ============== ===================")
            
            self.padded_stationary_img_tnsr = padded_stationary_img_tnsr
            self.padded_moving_img_tnsr = padded_moving_img_tnsr
            self.img_shape =  max_shapes_array
            return self.padded_stationary_img_tnsr, self.padded_moving_img_tnsr, self.stationary_img_voxel_dim, self.moving_img_voxel_dim, self.stationary_img_centre, self.moving_img_centre, self.img_shape;

        else:
            print(" ============= ============== ===================")
            print("Not padding as pad flag is false")
            stationary_img_shape = self.resampled_stationary_img.header.get_data_shape()
            moving_img_shape = self.resampled_moving_img.header.get_data_shape()

            stationary_img_np = np.array(self.resampled_stationary_img.dataobj)
            moving_img_np = np.array(self.resampled_moving_img.dataobj)
            max_shapes_array = [max(stationary_img_shape[0], moving_img_shape[0]), max(stationary_img_shape[1], moving_img_shape[1]), max(stationary_img_shape[2], moving_img_shape[2])]
      
            stationary_img_tnsr = torch.from_numpy(stationary_img_np)        
            moving_img_tnsr = torch.from_numpy(moving_img_np)

            print(" ============= ============== ===================")
            print(stationary_img_tnsr.shape)
            print(" ============= ============== ===================")
            print(moving_img_tnsr.shape)
            print(" ============= ============== ===================")

            print(stationary_img_tnsr.type())
            print(" ============= ============== ===================")
            print(moving_img_tnsr.type())
            print(" ============= ============== ===================")

            self.padded_stationary_img_tnsr = self.stationary_img_tnsr
            self.padded_moving_img_tnsr = self.moving_img_tnsr
            self.img_shape = max_shapes_array
            return self.padded_stationary_img_tnsr, self.padded_moving_img_tnsr, self.stationary_img_voxel_dim, self.moving_img_voxel_dim, self.stationary_img_centre, self.moving_img_centre, self.img_shape;


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
