
import os, glob
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
print(tf.__version__)
import nibabel as nb
import nilearn as nl
import voxelmorph as vxm
import nilearn.image
import neurite as ne


class InferenceDriver:

    def __init__(self, fixed_folder_path, moving_folder_path, warped_folder_path,
                model_path = "/Users/surajshashidhar/git/image_registration_data/bkp_trained_model/t2_mv_t1_fix_registration_weights/t2_mv_t1_fix_registration_weights.index", 
                downsample_shape=(128, 128, 128), batch_size=1,
                nb_features = [[32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 16]]):
        self.fixed_folder_path = fixed_folder_path
        self.moving_folder_path = moving_folder_path
        self.model_path = model_path
        self.downsample_shape = downsample_shape
        self.batch_size = batch_size
        self.nb_features = nb_features
        self.fixed_file_list = None
        self.moving_file_list = None
        self.fixed_data_np = None
        self.moving_data_np = None
        self.vxm_model = None


    def getFilenamesList(self):
        self.fixed_file_list = glob.glob(os.path.join(self.fixed_folder_path,'*'))
        self.moving_file_list = glob.glob(os.path.join(self.moving_folder_path,'*'))
        print("Number of files in fixed and moving folder are {} and {}".format(len(self.fixed_file_list), len(self.moving_file_list)))
        
        return ;


    def preprocessImages(self):
        self.fixed_data_np = np.zeros(shape=(len(self.fixed_file_list), self.downsample_shape[0], self.downsample_shape[1], self.downsample_shape[2]))
        self.moving_data_np = np.zeros(shape=(len(self.moving_file_list), self.downsample_shape[0], self.downsample_shape[1], self.downsample_shape[2]))

        target_shape = np.array(self.downsample_shape)
        new_resolution = [2,]*3
        new_affine = np.zeros((4,4))
        new_affine[:3,:3] = np.diag(new_resolution)
        new_affine[:3,3] = target_shape*new_resolution/2.0*-1
        new_affine[3,3] = 1.0

        for i in range(len(self.fixed_file_list)):
            fixed_file = self.fixed_file_list[i]
            moving_file = self.moving_file_list[i]
            fixed_inp_nb = nl.image.load_img(fixed_file)
            moving_inp_nb = nl.image.load_img(moving_file)

            interim_nb1 = nl.image.resample_img(fixed_inp_nb, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
            interim_nb2 = nl.image.resample_img(moving_inp_nb, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')

            fixed_np = interim_nb1.dataobj
            moving_np = interim_nb2.dataobj
            fixed_np = fixed_np/np.max(fixed_np)
            moving_np = moving_np/np.max(moving_np)

            self.fixed_data_np[i,:,:,:] = fixed_np
            self.moving_data_np[i,:,:,:] = moving_np

        return;


    def vxm_data_generator(self):
        """
        Generator that takes in data of size [N, H, W, D], and yields data for
        our custom vxm model. Note that we need to provide numpy data for each
        input, and each output.

        inputs:  moving [bs, H, W, D], fixed image [bs, H, W, D]
        outputs: moved image [bs, H, W, D, 1], zero-gradient [bs, H, W, D, 2]
        """

        # preliminary sizing
        vol_shape = self.fixed_data_np.shape[1:] # extract data shape
        print("Volume shape: {}".format(vol_shape))
        ndims = len(vol_shape)
        
        # prepare a zero array the size of the deformation
        # we'll explain this below
        zero_phi = np.zeros([self.batch_size, *vol_shape, ndims])
        
        while True:
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, 1]
            idx1 = np.random.randint(0, self.moving_data_np.shape[0], size=self.batch_size)
            moving_images = self.moving_data_np[idx1, ..., np.newaxis]
            idx2 = np.random.randint(0, self.fixed_data_np.shape[0], size=self.batch_size)
            fixed_images = self.fixed_data_np[idx2, ..., np.newaxis]
            inputs = [moving_images, fixed_images]
            
            # prepare outputs (the 'true' moved image):
            # of course, we don't have this, but we know we want to compare 
            # the resulting moved image with the fixed image. 
            # we also wish to penalize the deformation field. 
            outputs = [fixed_images, zero_phi]
            
            yield (inputs, outputs)


    def loadModel(self):
        nb_features = self.nb_features
        vol_shape = self.downsample_shape
        print("shape of volume: {}".format(vol_shape))
        print("Filters in Unet: {}".format(nb_features))
        vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=3)
        vxm_model.load_weights(self.model_path)
        self.vxm_model = vxm_model
        print(self.vxm_model.summary());
        return;


    def inferImages(self, inference_in_sample):
        inference_pred = self.vxm_model.predict(inference_in_sample)
        warped_images = inference_pred[0]
        deformation_field = inference_pred[1]


        for i in range(warped_images.shape[0]):
            tmp_img_np = warped_images[i, :, :, :, 0]
            tmp_deformation_field_np = deformation_field[i, :, :, :, :]

            warped_image_nb = nb.Nifti1Image(tmp_img_np, affine=np.eye(4))
            deformation_field_nb = nb.Nifti1Image(tmp_deformation_field_np, affine=np.eye(4))

            imt1 = os.path.basename(self.moving_file_list[i])
            imt1 = imt1[0:-7]

            warped_image_nb.to_filename(os.path.join(self.warped_folder_path, imt1 + "_warped_image" + ".nii.gz"))
            deformation_field_nb.to_filename(os.path.join(self.warped_folder_path,imt1 + "_deformation_field" + ".nii.gz"))


    def runInference(self):
        self.getFilenamesList()
        self.preprocessImages()
        inference_generator = self.vxm_data_generator()
        inference_in_sample, inference_out_sample = next(inference_generator)

        # visualize
        #images = [img[0, :, 64, :, 0] for img in inference_in_sample + inference_out_sample]
        #titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
        # ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);
        self.loadModel()
        self.inferImages(inference_in_sample)


if __name__ == "__main__":
    fixed_folder_path = "/Users/surajshashidhar/git/image_registration_data/t1_test"
    moving_folder_path = "/Users/surajshashidhar/git/image_registration_data/t2_test"
    warped_folder_path = "/Users/surajshashidhar/git/image_registration_data/warped_images"
    model_path = "/Users/surajshashidhar/git/image_registration_data/bkp_trained_model/t2_mv_t1_fix_registration_weights/t2_mv_t1_fix_registration_weights.index"
    inference_obj = InferenceDriver(fixed_folder_path, moving_folder_path, warped_folder_path)
    inference_obj.runInference()

