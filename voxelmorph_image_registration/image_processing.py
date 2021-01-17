import nibabel as nb
import nilearn as nl
import nilearn.image
import numpy as np
import glob
import os
import voxelmorph as vxm


class ImageProcessing:


    def __init__(self, fixed_image_folder, moving_image_folder, img_dim = (256, 256, 256), training_batch_size=3, 
                resampling_shape = (128,128,128) ):
        self.fixed_image_folder = fixed_image_folder
        self.moving_image_folder = moving_image_folder
        self.moving_file_list = None
        self.fixed_file_list = None
        self.img_dim = img_dim
        self.fixed_data_np = None
        self.moving_data_np = None
        self.training_batch_size = training_batch_size
        self.resamplng_shape = resampling_shape


    def getListofImages(self):
        fixed_file_list = glob.glob(os.path.join(self.fixed_image_folder,'*'))
        moving_file_list = glob.glob(os.path.join(self.moving_image_folder,'*'))
        print("Moving and Fixed files are: {}, {}".format(len(moving_file_list), len(fixed_file_list)))
        self.moving_file_list = moving_file_list
        self.fixed_file_list = fixed_file_list
        return;


    def readImagesfromListnew(self):
        self.fixed_data_np = np.zeros(shape=(len(self.fixed_file_list), self.resamplng_shape[0] , self.resamplng_shape[1] , self.resamplng_shape[2] ))
        self.moving_data_np = np.zeros(shape=(len(self.moving_file_list),  self.resamplng_shape[0] , self.resamplng_shape[1] , self.resamplng_shape[2]))
        ind = 0
        for fixed_img, moving_img in zip(self.fixed_file_list, self.moving_file_list):
            fixed_np = self.run3functions(fixed_img)
            moving_np = self.run3functions(moving_img)
            self.fixed_data_np[ind,:,:,:] = fixed_np
            self.moving_data_np[ind,:,:,:] = moving_np
            ind = ind + 1


    def readImagesfromList(self):
        self.fixed_data_np = np.zeros(shape=(len(self.fixed_file_list), self.resamplng_shape[0] , self.resamplng_shape[1] , self.resamplng_shape[2] ))
        self.moving_data_np = np.zeros(shape=(len(self.moving_file_list),  self.resamplng_shape[0] , self.resamplng_shape[1] , self.resamplng_shape[2]))
        ind = 0
        for fixed_img, moving_img in zip(self.fixed_file_list, self.moving_file_list):
            fixed_nb = nl.image.load_img(fixed_img)
            moving_nb = nl.image.load_img(moving_img)

            target_shape = np.array(self.resamplng_shape)
            new_resolution = [2,]*3
            new_affine = np.zeros((4,4))
            new_affine[:3,:3] = np.diag(new_resolution)
            new_affine[:3,3] = target_shape*new_resolution/2.*-1
            new_affine[3,3] = 1.0

            interim_nb1 = nl.image.resample_img(fixed_nb, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
            interim_nb2 = nl.image.resample_img(moving_nb, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')

            fixed_np = interim_nb1.dataobj
            moving_np = interim_nb2.dataobj

            fixed_np = fixed_np/np.max(fixed_np)
            moving_np = moving_np/np.max(moving_np)

            self.fixed_data_np[ind,:,:,:] = fixed_np
            self.moving_data_np[ind,:,:,:] = moving_np
            
            ind = ind + 1

        print("Shapes of fixed and moving images numpy arrays are: {}, {}".format(self.fixed_data_np.shape, self.moving_data_np.shape))

    
    def vxm_data_generator(self, batch_size=None):
        """
        Generator that takes in data of size [N, H, W, D], and yields data for
        our custom vxm model. Note that we need to provide numpy data for each
        input, and each output.

        inputs:  moving [bs, H, W, D], fixed image [bs, H, W, D]
        outputs: moved image [bs, H, W, D, 1], zero-gradient [bs, H, W, D, 2]
        """

        if (batch_size == None):
            batch_size = self.training_batch_size

        # preliminary sizing
        vol_shape = self.fixed_data_np.shape[1:] # extract data shape
        ndims = len(vol_shape)
        
        # prepare a zero array the size of the deformation
        # we'll explain this below
        zero_phi = np.zeros([batch_size, *vol_shape, ndims])
        
        while True:
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, 1]
            idx1 = np.random.randint(0, self.moving_data_np.shape[0], size=batch_size)
            moving_images = self.moving_data_np[idx1, ..., np.newaxis]
            idx2 = np.random.randint(0, self.fixed_data_np.shape[0], size=batch_size)
            fixed_images = self.fixed_data_np[idx2, ..., np.newaxis]
            inputs = [moving_images, fixed_images]
            
            # prepare outputs (the 'true' moved image):
            # of course, we don't have this, but we know we want to compare 
            # the resulting moved image with the fixed image. 
            # we also wish to penalize the deformation field. 
            outputs = [fixed_images, zero_phi]
            
            yield (inputs, outputs)


    def load_3D(self, name):
        model_np = np.zeros(shape=self.resamplng_shape)
        X_nb = nb.load(name)
        X_np = X_nb.dataobj
        x_dim, y_dim, z_dim = X_np.shape
        x_ltail = (self.resamplng_shape[0] - x_dim)//2 
        y_ltail = (self.resamplng_shape[1] - y_dim)//2
        z_ltail = (self.resamplng_shape[2] - z_dim)//2

        x_rtail = self.resamplng_shape[0] - x_ltail - 1
        y_rtail = self.resamplng_shape[1] - y_ltail - 1
        z_rtail = self.resamplng_shape[2] - z_ltail - 1
        #print("Oreintation: {}".format(nb.aff2axcodes(X_nb.affine)))
        #model_np[:, :, :] = X_np[42:202, 32:224, 16:240]
        model_np[x_ltail:x_rtail, y_ltail:y_rtail, z_ltail:z_rtail] = X_np[:, :, :]
        #model_np = np.reshape(model_np, (1,)+ model_np.shape)
        return model_np


    def imgnorm(self, N_I,index1=0.0001,index2=0.0001):
        I_sort = np.sort(N_I.flatten())
        I_min = I_sort[int(index1*len(I_sort))]
        I_max = I_sort[-int(index2*len(I_sort))]
        N_I =1.0*(N_I-I_min)/(I_max-I_min)
        N_I[N_I>1.0]=1.0
        N_I[N_I<0.0]=0.0
        N_I2 = N_I.astype(np.float32)
        return N_I2


    def Norm_Zscore(self, img):
        img= (img-np.mean(img))/np.std(img) 
        return img


    def run3functions(self, fp):
        myimg = self.load_3D(fp)
        #myimg1 = self.Norm_Zscore(self.imgnorm(myimg))
        myimg2 = self.imgnorm(myimg)
        return myimg2


    def generateDataset(self, batch_size):
        generator = self.vxm_data_generator( batch_size=batch_size)
        in_sample, out_sample = next(generator)
        return generator;




"""
x_ltail,x_dim, x_rtail, y_ltail, y_dim, y_rtail, z_ltail, z_dim, z_rtail : 18, 91, 109, 9, 109, 118, 18, 91, 109
Traceback (most recent call last):
  File "driver.py", line 78, in <module>
    ipr_obj.readImagesfromListnew()
  File "/project/shashidh/voxelmorph_image_registration/image_processing.py", line 40, in readImagesfromListnew
    fixed_np = self.run3functions(fixed_img)
  File "/project/shashidh/voxelmorph_image_registration/image_processing.py", line 156, in run3functions
    myimg = self.load_3D(fp)
  File "/project/shashidh/voxelmorph_image_registration/image_processing.py", line 134, in load_3D
    model_np[:, :, :] = X_np[x_ltail:x_rtail, y_ltail:y_rtail, z_ltail:z_rtail]
ValueError: could not broadcast input array from shape (73,100,73) into shape (128,128,128)

"""
        
