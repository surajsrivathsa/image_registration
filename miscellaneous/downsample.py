import nibabel as nb
import numpy as np
import glob
import os
from skimage.transform import resize


def getListofImages(folderpath):
        file_list = glob.glob(os.path.join(folderpath,'*'))
        print("file list size is: {}".format(len(file_list)))
        file_list = file_list
        return file_list;


def downsampleImages(file_list):
    for fl in file_list:        
        fl_name = os.path.splitext(os.path.splitext(os.path.basename(fl))[0])[0]
        img = nb.load(fl)
        header_info = img.header
        # change the voxel dimensions to [2,2,2]
        header_info['pixdim'][1:4]  = [1,1,1]
        data = np.array(img.get_fdata())
        imgres = resize(data, (128,128,128),3)
        x = nb.Nifti1Image(imgres, img.affine, header_info)
        nb.save(x, "/Users/surajshashidhar/Downloads/Voxelmorph_dataset_downsampled/" + fl_name + ".nii.gz")
        print("resampled {}".format(fl_name))
    
    return;

file_list = getListofImages("/Users/surajshashidhar/Downloads/Voxel_morph_dataset")
downsampleImages(file_list)


"""
print("Hello world")
img = nb.load('/Users/surajshashidhar/Downloads/Voxel_morph_dataset/IXI002-Guys-0828-T1.nii.gz')
header_info = img.header
print()
print(header_info)
print("========= =========== ============= ==========")
print()
# change the voxel dimensions to [2,2,2]
header_info['pixdim'][1:4]  = [1,1,1]
data = np.array(img.get_fdata())
imgres = resize(data, (128,128,128),3)
print(imgres.shape)
print("========= =========== ============= ==========")
print()
x = nb.Nifti1Image(imgres, img.affine, header_info)
print(x.header)
print("========= =========== ============= ==========")
print()
nb.save(x, '/Users/surajshashidhar/Downloads/IXI002-Guys-0828-T1_downsampled.nii.gz')

Fixed file list: ['/project/shashidh/test_fixed_images/IXI034-HH-1260-T1.nii.gz', 
'/project/shashidh/test_fixed_images/IXI178-Guys-0778-T1.nii.gz', 
'/project/shashidh/test_fixed_images/IXI434-IOP-1010-T1_norm.nii.gz', 
'/project/shashidh/test_fixed_images/IXI462-IOP-1042-T1_norm.nii.gz', 
'/project/shashidh/test_fixed_images/IXI599-HH-2659-T1_norm.nii.gz']

Moving file list: ['/project/shashidh/test_moving_images/IXI002-Guys-0828-T1.nii.gz', 
'/project/shashidh/test_moving_images/IXI035-IOP-0873-T1.nii.gz', 
'/project/shashidh/test_moving_images/IXI391-Guys-0934-T1_norm.nii.gz', 
'/project/shashidh/test_moving_images/IXI525-HH-2413-T1_norm.nii.gz', 
'/project/shashidh/test_moving_images/IXI546-HH-2450-T1_norm.nii.gz']
"""