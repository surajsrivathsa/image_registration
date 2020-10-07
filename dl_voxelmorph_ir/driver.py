import os, sys
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
print("Tensorflow version: {}".format(tf.__version__))
import voxelmorph as vxm
import neurite as ne
import nibabel as nb
import nibabel.processing as nbp

vol_shape = (224, 320, 320)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]]

vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=2);

vxm_model.load_weights('/Users/surajshashidhar/git/image_registration/brain_3d.h5')
t1_fn= "/Users/surajshashidhar/git/image_registration/T1_images/IXI002-Guys-0828-T1.nii.gz"
t2_fn= "/Users/surajshashidhar/git/image_registration/T1_images/IXI030-Guys-0708-T1.nii.gz"

def preprocess_nifti_images( t1_fn, t2_fn, resampling_size = [1,1,1]):
   # read from T1 and T2 paths
        orig_nii_t1 = nb.load(t1_fn)
        orig_nii_t2 = nb.load(t2_fn)

        #print(orig_nii_t1.header)
        #print(orig_nii_t2.header)

        print(orig_nii_t1.header["pixdim"])
        print(orig_nii_t2.header["pixdim"])
        print(orig_nii_t1.header["qoffset_x"])
        print(orig_nii_t1.header["qoffset_y"])
        print(orig_nii_t1.header["qoffset_z"])
        print(orig_nii_t2.header["qoffset_x"])
        print(orig_nii_t2.header["qoffset_y"])
        print(orig_nii_t2.header["qoffset_z"])

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
               
        #t1_img_np = np.array(canonical_img_1.dataobj)
        #t2_img_np = np.array(canonical_img_2.dataobj)

        #t1_img_np = np.zeros((256, 288, 288))
        #t2_img_np = np.zeros((256, 288, 288))

        t1_img_np = np.zeros((224, 320, 320))
        t2_img_np = np.zeros((224, 320, 320))

        t1_img_np[:ci1_shape[0], :ci1_shape[1], :ci1_shape[2]] = canonical_img_1.dataobj
        t2_img_np[:ci2_shape[0], :ci2_shape[1], :ci2_shape[2]] = canonical_img_2.dataobj

        print(" ============= ============== ===================")
        print("Max intensity of T1: {}".format(np.max(t1_img_np)))
        print(" ============= ============== ===================")
        print("Max intensity of T2: {}".format(np.max(t2_img_np)))

        t1_img_np = t1_img_np/(np.max(t1_img_np) * 1.0)
        t2_img_np = t2_img_np/(np.max(t2_img_np) * 1.0)

        print(" ============= ============== ===================")
        print("Padded numpy array stationary image shape: {}".format(t1_img_np.shape))
        print(" ============= ============== ===================")
        print("Padded numpy array moving image shape: {}".format(t2_img_np.shape))
        print(" ============= ============== ===================")


        return t1_img_np, t2_img_np, canonical_img_1_voxel_dim, canonical_img_2_voxel_dim, canonical_img_1_centre, canonical_img_2_centre, max_shapes_array;


t1_img_np, t2_img_np, canonical_img_1_voxel_dim, canonical_img_2_voxel_dim, canonical_img_1_centre, canonical_img_2_centre, img_shape = preprocess_nifti_images(t1_fn=t1_fn, t2_fn=t2_fn);
val_volume_1 = t1_img_np
val_volume_2 = t2_img_np
val_input = [
    val_volume_1[np.newaxis, ..., np.newaxis],
    val_volume_2[np.newaxis, ..., np.newaxis]
]

val_pred = vxm_model.predict(val_input);
moved_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]

mid_slices_fixed = [np.take(val_volume_2, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_moving = [np.take(val_volume_1, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)

ne.plot.slices(mid_slices_fixed + mid_slices_moving + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[3, 3]);