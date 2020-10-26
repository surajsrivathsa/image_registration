
import tensorflow as tf
import voxelmorph as vxm

class Model:

    def __init__(self, train_generator, epochs=2, steps_per_epoch=1, verbose=2, vol_shape = (256, 256, 256), 
                model_save_path = "/project/shashidh/voxelmorph_models", 
                model_weights_save_path = "/project/shashidh/voxelmorph_model_weight" ):
        self.nb_features = [[32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 16]]
        self.vol_shape = vol_shape
        self.int_steps = 3
        self.vxm_model = None
        self.train_generator = train_generator
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.hist = None
        self.model_save_path = model_save_path
        self.model_weights_save_path = model_weights_save_path


    def buildModel(self):
        # voxelmorph model
        vxm_model = vxm.networks.VxmDense(self.vol_shape, self.nb_features, int_steps=self.int_steps)
        
        # losses and loss weights
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
        loss_weights = [1, 1]

        vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
        self.vxm_model = vxm_model
        return;


    def printModelSummary(self):
        print(self.vxm_model.summary())


    def trainModel(self):
        hist = self.vxm_model.fit_generator(self.train_generator, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, verbose=self.verbose);
        self.hist = None


    def saveModelandWeights(self):
        self.vxm_model.save(self.model_save_path)
        self.vxm_model.save_weights(self.model_weights_save_path)

