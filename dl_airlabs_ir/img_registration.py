import torch
import sys
import os
import time
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import nibabel.processing as nbp
import img_processing as ipr
import dl_airlabs_constants as const
sys.path.insert(0, const.FILEPATH_AIRLABS_LIB)
import airlab as al

class ImageRegistrationUtils:
    def __init__(self, preprocessed_stationary_img_tnsr, preprocessed_moving_img_tnsr, preprocessed_stationary_img_voxel_dim,
    preprocessed_moving_img_voxel_dim, preprocessed_stationary_img_centre, preprocessed_moving_img_centre,
    img_shape, device, loss_fnc = const.LOSS_FNC, logging_flag=const.LOGGING_FLAG, log = None):
        self.preprocessed_stationary_img_tnsr = preprocessed_stationary_img_tnsr
        self.preprocessed_moving_img_tnsr = preprocessed_moving_img_tnsr
        self.preprocessed_stationary_img_voxel_dim = preprocessed_stationary_img_voxel_dim
        self.preprocessed_moving_img_voxel_dim = preprocessed_moving_img_voxel_dim
        self.preprocessed_stationary_img_centre = preprocessed_stationary_img_centre
        self.preprocessed_moving_img_centre = preprocessed_moving_img_centre
        self.img_shape = img_shape
        self.device = device
        self.loss_fnc = loss_fnc 
        self.affine_transformation_matrix = None
        self.affine_transformation_object = None
        self.displacement = None
        self.logging_flag = logging_flag
        self.log = log

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
    


    def three_dim_affine_reg(self):
        start = time.time()

        # set the used data type
        dtype = torch.float32
        # set the device for the computaion to CPU
        #device = torch.device("cpu")
        #device = torch.device("cuda:0")
        device = self.device

        # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
        # Here, the GPU with the index 0 is used.
        # device = th.device("cuda:0")
        
        #Creating the airlabs image objects for registration
        new_stationary_img_tnsr = self.preprocessed_stationary_img_tnsr.to(device=device)
        new_moving_img_tnsr = self.preprocessed_moving_img_tnsr.to(device=device)
        fixed_image = al.Image(new_stationary_img_tnsr, self.img_shape, self.preprocessed_stationary_img_voxel_dim, self.preprocessed_stationary_img_centre)
        moving_image = al.Image(new_moving_img_tnsr, self.img_shape, self.preprocessed_moving_img_voxel_dim, self.preprocessed_moving_img_centre)
        

        # printing image properties
        print(" ============= fixed image size, spacing, origin and datatype ===================")
        print(fixed_image.size)
        print(fixed_image.spacing)
        print(fixed_image.origin)
        print(fixed_image.dtype)
        print(" ============= moving image size, spacing, origin and datatype ===================")
        print(moving_image.size)
        print(moving_image.spacing)
        print(moving_image.origin)
        print(moving_image.dtype)
        print(" ============= ============== ===================")

        # create pairwise registration object
        registration = al.PairwiseRegistration()

        # choose the affine transformation model
        print("Using Affine transformation")
        print(" ============= ============== ===================")
        transformation = al.transformation.pairwise.AffineTransformation(moving_image, opt_cm=True)
        transformation.init_translation(fixed_image)
        registration.set_transformation(transformation)

        
        # choose the Mean Squared Error as image loss
        if(self.loss_fnc == "MSE"):
            print("Using Mean squared error loss")
            image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
        elif(self.loss_fnc == "MI"):
            print("Using Mutual information loss")
            image_loss = al.loss.pairwise.MI(fixed_image, moving_image,bins=20, sigma=3)
        elif(self.loss_fnc == "CC"):
            print("Using Cross corelation loss")
            image_loss = al.loss.pairwise.NCC(fixed_image, moving_image)
        else:
            print("No valid option chosen among MSE/NCC/NMI, using MSE as default")
            image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

        registration.set_image_loss([image_loss])

        # choose the Adam optimizer to minimize the objective
        optimizer = torch.optim.Adam(transformation.parameters(), lr=0.1)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(const.ITERATIONS)

        # start the registration
        registration.start()

        # set the intensities for the visualisation
        fixed_image.image = 1 - fixed_image.image
        moving_image.image = 1 - moving_image.image

        # warp the moving image with the final transformation result
        displacement = transformation.get_displacement()
        warped_image = al.transformation.utils.warp_image(moving_image, displacement)

        end = time.time()

        print(" ============= ============== ===================")

        print("Registration done in: ", end - start, " s")
        print("Result parameters:")
        transformation.print()
        print(" ============= ============== ===================")
        print(transformation.transformation_matrix)
        print(" ============= ============== ===================")

        # plot the results - commented out as it pops open a window
        
        plt.subplot(131)
        plt.imshow(fixed_image.numpy()[90, :, :], cmap='gray')
        plt.title('Fixed Image Slice')

        plt.subplot(132)
        plt.imshow(moving_image.numpy()[90, :, :], cmap='gray')
        plt.title('Moving Image Slice')   

        plt.subplot(133)
        plt.imshow(warped_image.numpy()[16, :, :], cmap='gray')
        plt.title('Warped Moving Image Slice')   
        plt.show()
        

        self.affine_transformation_matrix = transformation.transformation_matrix
        self.affine_transformation_object = transformation
        self.displacement = displacement

        return warped_image, transformation, displacement;
