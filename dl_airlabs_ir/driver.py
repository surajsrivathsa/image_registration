import torch
import sys
import os
import time
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nb
import numpy as np
import nibabel.processing as nbp
import img_processing as ipr
import img_registration as ireg
sys.path.insert(0, '/Users/surajshashidhar/git/airlab')
import airlab as al

print("Hello world")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

stationary_image_file_path = '/Users/surajshashidhar/git/image_registration/T1_images/IXI002-Guys-0828-T1.nii.gz'
moving_image_file_path = '/Users/surajshashidhar/git/image_registration/T2_images/IXI002-Guys-0828-T2.nii.gz'
output_warped_image_file_path = "/Users/surajshashidhar/git/image_registration/warped_image1.nii.gz"

print()
print(" ============= preprocessing started ===================")
img_prcs_obj = ipr.ImageprocessingUtils(stationary_image_file_path=stationary_image_file_path, moving_image_file_path=moving_image_file_path, output_warped_image_file_path = output_warped_image_file_path)
orig_nii_stationary, orig_nii_moving = img_prcs_obj.read_input_images();
canonical_img_1, canonical_img_2 = img_prcs_obj.reorient_images();
resampled_stationary_img, resampled_moving_img = img_prcs_obj.resample_image();
preprocessed_stationary_img_tnsr, preprocessed_moving_img_tnsr, preprocessed_stationary_img_voxel_dim, preprocessed_moving_img_voxel_dim, preprocessed_stationary_img_centre, preprocessed_moving_img_centre, img_shape = img_prcs_obj.convert_nifti_to_tensor();

print()
print(" ============= preprocessing completed ===================")
print()
print(" ============= starting registration ===================")
img_regs_obj = ireg.ImageRegistrationUtils(preprocessed_stationary_img_tnsr, preprocessed_moving_img_tnsr, preprocessed_stationary_img_voxel_dim, preprocessed_moving_img_voxel_dim, preprocessed_stationary_img_centre, preprocessed_moving_img_centre, img_shape, device)
warped_img_tnsr, transformation, displacement = img_regs_obj.three_dim_affine_reg();
print(" ============= registration ended===================")
print()
print(" ============= starting post processing and saving warped image to disk ===================")
print()
warped_nifti_img = img_prcs_obj.convert_tensor_to_nifti(warped_img_tnsr, transformation, displacement)
img_prcs_obj.save_warped_image(warped_nifti_img)
print()
print(" ============= warped image saved to path ===================")
sys.exit(0)


def preprocess_nifti_images(resampling_size = [1,1,1]):
   # read from T1 and T2 paths
        orig_nii_t1 = nb.load(stationary_image_file_path)
        orig_nii_t2 = nb.load(moving_image_file_path)

        orig_nii_t1_voxel_dim = orig_nii_t1.header["pixdim"][1:4]
        orig_nii_t2_voxel_dim = orig_nii_t2.header["pixdim"][1:4]
        orig_nii_t1_centre = [float(orig_nii_t1.header["qoffset_x"]), float(orig_nii_t1.header["qoffset_y"]), float(orig_nii_t1.header["qoffset_z"])]
        orig_nii_t2_centre = [float(orig_nii_t2.header["qoffset_x"]), float(orig_nii_t2.header["qoffset_y"]), float(orig_nii_t2.header["qoffset_z"])]


        print(" ============= ============== ===================")
        print("Image 1 voxel resolution before resampling: {}".format(orig_nii_t1_voxel_dim))
        print(" ============= ============== ===================")
        print("Image 2 voxel resolution before resampling: {}".format(orig_nii_t2_voxel_dim))
        print(" ============= ============== ===================")
        print("Image 1 centre before resampling: {}".format(orig_nii_t1_centre))
        print(" ============= ============== ===================")
        print("Image 2 centre before resampling: {}".format(orig_nii_t2_centre))
        

        print("original t1 affine: {}".format(orig_nii_t1.affine))
        print(" ============= ============== ===================")
        print("original t2 affine: {}".format(orig_nii_t2.affine))
        print(" ============= ============== ===================")
        print("original t1 Orientation: {}".format(nb.aff2axcodes(orig_nii_t1.affine)))
        print(" ============= ============== ===================")
        print("original t2 Orientation: {}".format(nb.aff2axcodes(orig_nii_t2.affine)))

        canonical_img_1 = nb.as_closest_canonical(orig_nii_t1)
        print(" ============= ============== ===================")
        print("orientation changed  t1 affine: {}".format(canonical_img_1.affine))
        print(" ============= ============== ===================")
        print("orientation changed  t1 : {}".format(nb.aff2axcodes(canonical_img_1.affine)))
        print(" ============= ============== ===================")
        canonical_img_2 = nb.as_closest_canonical(orig_nii_t2)
        print(" ============= ============== ===================")
        print("orientation changed  t2 affine: {}".format(canonical_img_2.affine))
        print(" ============= ============== ===================")
        print("orientation changed  t1 : {}".format(nb.aff2axcodes(canonical_img_2.affine)))

        resampled_voxel_size = resampling_size
        canonical_img_1 = nb.processing.resample_to_output(canonical_img_1,voxel_sizes=resampled_voxel_size)
        canonical_img_2 = nb.processing.resample_to_output(canonical_img_2,voxel_sizes=resampled_voxel_size)

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
               
        t1_img_np = np.array(canonical_img_1.dataobj)
        t2_img_np = np.array(canonical_img_2.dataobj)

        t1_img_tnsr = torch.from_numpy(t1_img_np)        
        t2_img_tnsr = torch.from_numpy(t2_img_np)  

        padded_t1_img_tnsr = torch.zeros(max_shapes[0], max_shapes[1], max_shapes[2])
        padded_t2_img_tnsr = torch.zeros(max_shapes[0], max_shapes[1], max_shapes[2])

        padded_t1_img_tnsr[:t1_img_tnsr.shape[0], :t1_img_tnsr.shape[1], :t1_img_tnsr.shape[2]] = t1_img_tnsr
        padded_t2_img_tnsr[:t2_img_tnsr.shape[0], :t2_img_tnsr.shape[1], :t2_img_tnsr.shape[2]] = t2_img_tnsr

        padded_t1_img_tnsr = padded_t1_img_tnsr.type(torch.FloatTensor)
        padded_t2_img_tnsr = padded_t2_img_tnsr.type(torch.FloatTensor)

        print(padded_t1_img_tnsr.shape)
        print(padded_t2_img_tnsr.shape)

        print(padded_t1_img_tnsr.type())
        print(padded_t2_img_tnsr.type())

        #print(padded_t1_img_tnsr.header)
        #print(padded_t1_img_tnsr.header)

        return padded_t1_img_tnsr, padded_t2_img_tnsr, canonical_img_1_voxel_dim, canonical_img_2_voxel_dim, canonical_img_1_centre, canonical_img_2_centre, max_shapes_array;


preprocessed_stationary_img_tnsr, preprocessed_moving_img_tnsr, canonical_stationary_img_voxel_dim, canonical_moving_img_voxel_dim, canonical_stationary_img_centre, canonical_moving_img_centre, img_shape = preprocess_nifti_images();


def three_dim_affine_reg(stationary_img_tnsr=preprocessed_stationary_img_tnsr, moving_img_tnsr=preprocessed_moving_img_tnsr, 
                         stationary_img_voxel_size = canonical_stationary_img_voxel_dim, moving_img_voxel_size = canonical_moving_img_voxel_dim, 
                         stationary_img_centre = canonical_stationary_img_centre, moving_img_centre = canonical_moving_img_centre, 
                         img_shape = img_shape, device=device, loss_fnc = "MSE", mode = "normal"):
    start = time.time()

    # set the used data type
    dtype = torch.float32
    # set the device for the computaion to CPU
    #device = torch.device("cpu")
    #device = torch.device("cuda:0")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # create 3D image volume with two objects
    object_shift = 50
    scaler = 2

    if(mode == "normal"):
      fixed_image = torch.zeros(256, 256, 256).to(device=device)
      fixed_image[16:128, 16:128, 16:128] = 1.0
      fixed_image = al.Image(fixed_image, [256, 256, 256], [1,1,1], [0,0,0])
      
      moving_image = torch.zeros(256, 256, 256).to(device=device)
      moving_image[(16+object_shift):(64+object_shift) * scaler, (16+object_shift):(64+object_shift) * scaler, (16+object_shift):(64+object_shift) * scaler] = 1.0  
      moving_image = al.Image(moving_image, [256, 256, 256], [1,1,1], [0,0,0])
      
    
    else:
      new_stationary_img_tnsr = stationary_img_tnsr.to(device=device)
      new_moving_img_tnsr = moving_img_tnsr.to(device=device)
      fixed_image = al.Image(new_stationary_img_tnsr, img_shape, stationary_img_voxel_size, stationary_img_centre)
      moving_image = al.Image(new_moving_img_tnsr, img_shape, moving_img_voxel_size, moving_img_centre)
      

    # printing image properties
    print(fixed_image.size)
    print(fixed_image.spacing)
    print(fixed_image.origin)
    print(fixed_image.dtype)

    print(moving_image.size)
    print(moving_image.spacing)
    print(moving_image.origin)
    print(moving_image.dtype)

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    print("Using Affine transformation")
    transformation = al.transformation.pairwise.AffineTransformation(moving_image, opt_cm=True)
    transformation.init_translation(fixed_image)
    registration.set_transformation(transformation)

    
    # choose the Mean Squared Error as image loss
    if(loss_fnc == "MSE"):
      print("Using Mean squared error loss")
      image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
    elif(loss_fnc == "MI"):
      print("Using Mutual information loss")
      image_loss = al.loss.pairwise.MI(fixed_image, moving_image,bins=20, sigma=3)
    elif(loss_fnc == "CC"):
      print("Using Cross corelation loss")
      image_loss = al.loss.pairwise.NCC(fixed_image, moving_image)
    else:
      print("No valid option chosen among MSE/NCC/NMI, using MSE as default")
      image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
    #image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.1)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(3)

    # start the registration
    registration.start()

    # set the intensities for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start, " s")
    print("Result parameters:")
    transformation.print()
    print("=================================================================")
    print(transformation.transformation_matrix)
    print("=================================================================")

    # plot the results
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

    return warped_image, transformation, displacement;

warped_img_tnsr, transformation, displacement = three_dim_affine_reg(stationary_img_tnsr=preprocessed_stationary_img_tnsr, moving_img_tnsr=preprocessed_moving_img_tnsr,loss_fnc="MSE",mode="mri")

print("=============== transformation matrix before ==============================")
print(transformation.transformation_matrix)

warped_img_np = warped_img_tnsr.numpy();
affine_transformation_matrix_np = transformation.transformation_matrix.detach().cpu().numpy()
final_transformation_matrix = np.identity(4)
final_transformation_matrix[0:3, :] = affine_transformation_matrix_np[: , :]

print("============== transformation matrix after ==================")
print(final_transformation_matrix)

warped_nifti_img = nb.Nifti1Image(warped_img_np, affine=final_transformation_matrix)
warped_nifti_img.to_filename("/Users/surajshashidhar/git/image_registration/warped_image.nii.gz")



"""
stationary_image_nb = nb.load(stationary_image_file_path)
moving_image_nb = nb.load(moving_image_file_path)
stationary_image_nb_data_1 = stationary_image_nb.get_fdata()
moving_image_nb_data_2 = moving_image_nb.get_fdata()


print(stationary_image_nb.affine)
print(" ============= ============== ===================")
print(moving_image_nb.affine)
print(" ============= ============== ===================")
print(nb.aff2axcodes(stationary_image_nb.affine))
print(" ============= ============== ===================")
print(nb.aff2axcodes(moving_image_nb.affine))

resampled_voxel_size = [1,1,1]
stationary_image_nb = nb.processing.resample_to_output(stationary_image_nb,voxel_sizes=resampled_voxel_size)
moving_image_nb = nb.processing.resample_to_output(moving_image_nb,voxel_sizes=resampled_voxel_size)

canonical_stationary_img = nb.as_closest_canonical(stationary_image_nb)
print(" ============= ============== ===================")
print(canonical_stationary_img.affine)
print(" ============= ============== ===================")
print(nb.aff2axcodes(canonical_stationary_img.affine))
print(" ============= ============== ===================")
canonical_moving_img = nb.as_closest_canonical(moving_image_nb)
print(" ============= ============== ===================")
print(canonical_moving_img.affine)
print(" ============= ============== ===================")
print(nb.aff2axcodes(canonical_moving_img.affine))

print(canonical_stationary_img.header.get_data_shape())
print(canonical_moving_img.header.get_data_shape())
"""