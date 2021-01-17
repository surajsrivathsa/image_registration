import os
import tensorflow as tf
import voxelmorph as vxm

class Model:

    def __init__(self, train_generator, epochs=2, steps_per_epoch=1, verbose=2, vol_shape = (256, 256, 256), 
                model_save_path = "/project/shashidh/voxelmorph_models", 
                model_weights_save_path = "/project/shashidh/voxelmorph_model_weight",
                similarity_loss_type = vxm.losses.NCC().loss, similarity_loss_weight = 1,
                regularizer_loss_type = vxm.losses.Grad("l2").loss, regularizer_loss_weight = 1, 
                use_pretrained_model = False , pretrained_model_path = "/project/shashidh/brain_3d.h5"):
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
        self.similarity_loss_type = similarity_loss_type
        self.regularizer_loss_type = regularizer_loss_type
        self.similarity_loss_weight = similarity_loss_weight
        self.regularizer_loss_weight = regularizer_loss_weight
        self.save_filename = os.path.join(model_save_path, '{epoch:04d}.h5')
        self.use_pretrained_model = use_pretrained_model
        self.pretrained_model_path = pretrained_model_path


    def buildModel(self):
        if (self.use_pretrained_model):
            nb_features = [
                [16, 32, 32, 32],
                [32, 32, 32, 32, 32, 16, 16]]
            vxm_model = vxm.networks.VxmDense(self.vol_shape, nb_features, int_steps=self.int_steps)
            vxm_model.load_weights(self.pretrained_model_path)
            losses = [self.similarity_loss_type, self.regularizer_loss_type]
            loss_weights = [self.similarity_loss_weight, self.regularizer_loss_weight]

            vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
            self.vxm_model = vxm_model
        else:
            vxm_model = vxm.networks.VxmDense(self.vol_shape, self.nb_features, int_steps=self.int_steps)

            losses = [self.similarity_loss_type, self.regularizer_loss_type]
            loss_weights = [self.similarity_loss_weight, self.regularizer_loss_weight]

            vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
            self.vxm_model = vxm_model
        return;


    def printModelSummary(self):
        print(self.vxm_model.summary())


    def trainModel(self):
        save_callback = tf.keras.callbacks.ModelCheckpoint(self.save_filename)
        hist = self.vxm_model.fit_generator(self.train_generator, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, callbacks=[save_callback], verbose=self.verbose);
        self.hist = hist


    def saveModelandWeights(self):
        os.mkdir(self.model_weights_save_path)
        self.vxm_model.save(self.model_save_path)
        self.vxm_model.save_weights(self.model_weights_save_path)


