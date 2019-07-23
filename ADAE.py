"""
Code reproduce for Anomaly Detection with Adversarial Dual Autoencoders

https://arxiv.org/pdf/1902.06924.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.python.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Concatenate
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU
from tensorflow.python.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np


def vae_z(latent_dim=2, epsilon_std=1.0):
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def f(z_mean, z_log_var):
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        return z
    return f


def lr_scheduler(initial_lr=1e-3, decay_factor=0.75, step_size=5, min_lr=1e-5):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        if lr > min_lr:
            return lr
        return min_lr

    return LearningRateScheduler(schedule, verbose=1)


class Autoencoder():
    def __init__(self, image_size, intermediate_dim=128, latent_dim=32):
        self.image_size = image_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.input = Input(shape=self.image_size)
        self.params = {
            'enc_filters': [16, 32, 48, 16, 32],
            'enc_kernels': [3, 3, 3, 4, 1],
            'enc_strides': [2, 2, 2, 1, 1],
            'dec_filters': [16, 16, 16, 32],
            'dec_kernels': [4, 3, 3, 3],
            'dec_strides': [1, 2, 2, 2],
        }
        self.reconstruction_shape = []

    def build_encoder(self, filters=[64, 128, 192, 64, 128], kernels=[3, 3, 3, 4, 1], strides=[2, 2, 2, 1, 1]):
        def f(x):
            for num_filter, kernel, stride in zip(filters, kernels, strides):
                x = Conv2D(
                    num_filter,
                    kernel_size=kernel,
                    padding='same',
                    activation='relu',
                    strides=stride)(x)
                x = BatchNormalization()(x)
                self.reconstruction_shape += [x.get_shape().as_list()]
            return x
        return f

    def build_decoder(self, filters=[64, 64, 64, 32], kernels=[4, 3, 3, 3], strides=[1, 2, 2, 2]):
        def f(x):
            for i, (num_filter, kernel, stride) in enumerate(zip(filters, kernels, strides)):
                x = Conv2DTranspose(num_filter,
                                    kernel_size=kernel,
                                    padding='same',
                                    strides=stride)(x)
                x = LeakyReLU(alpha=0.3)(x)
                x = BatchNormalization()(x)
            decoder = Conv2D(self.image_size[-1],
                             kernel_size=2,
                             padding='same',
                             activation='tanh')(x)
            return decoder
        return f

    def build_vae_model(self):
        hidden, z_mean, z_log_var = self.build_encoder(
            filters=self.params['enc_filters'], kernels=self.params['enc_kernels'], strides=self.params['enc_strides'])(self.input)
        z = vae_z(latent_dim=self.latent_dim)(z_mean, z_log_var)
        # z = Concatenate()([z_mean, z_log_var, z])
        dec = self.build_decoder(
            filters=self.params['dec_filters'], kernels=self.params['dec_kernels'], strides=self.params['dec_strides'])(z)
        # instantiate VAE model
        vae = Model(self.input, dec)
        # Compute VAE loss
        xent_loss = self.image_size[0] * self.image_size[1] * metrics.binary_crossentropy(
            K.flatten(self.input),
            K.flatten(dec))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        vae.add_loss(vae_loss)

        return vae

    def build_model(self):
        hidden = self.build_encoder(
            filters=self.params['enc_filters'], kernels=self.params['enc_kernels'], strides=self.params['enc_strides'])(self.input)
        dec = self.build_decoder(
            filters=self.params['dec_filters'], kernels=self.params['dec_kernels'], strides=self.params['dec_strides'])(hidden)
        # instantiate VAE model
        vae = Model(self.input, dec)
        return vae


class ADAE(object):
    def __init__(self, image_size=(28, 28, 1), latent_dim=100):
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.input = Input(shape=image_size)
        # Build the generator
        self.generator = Autoencoder(image_size=image_size).build_model()
        # Build and compile the discriminator
        self.discriminator = Autoencoder(image_size=image_size).build_model()

        self.gx = self.generator(self.input)
        self.dx = self.discriminator(self.input)
        self.dgx = self.discriminator(self.gx)

        self.d_loss = Lambda(lambda x: K.mean(K.mean(K.mean(K.abs(x[0] - x[1]), axis=1), axis=1), axis=1) -
                             K.mean(K.mean(K.mean(K.abs(x[2] - x[3]), axis=1), axis=1), axis=1), name='d_loss')([self.input, self.dx, self.gx, self.dgx])
        self.g_loss = Lambda(lambda x: K.mean(K.mean(K.mean(K.abs(x[0] - x[1]), axis=1), axis=1), axis=1) +
                             K.mean(K.mean(K.mean(K.abs(x[1] - x[2]), axis=1), axis=1), axis=1), name='g_loss')([self.input, self.gx, self.dgx])

        self.model = Model(inputs=[self.input], outputs=[self.g_loss, self.d_loss])
        self.model.summary()
        # self.generator.summary()
        # self.discriminator.summary()

    def get_anomaly_score(self):
        """ Compute the anomaly score. Call it after training. """
        score_out = Lambda(lambda x:
                           K.mean(K.mean(K.mean((x[0] - x[1]) ** 2, axis=1), axis=1), axis=1)
                           )([self.model.inputs[0], self.model.layers[2](self.model.layers[1](self.model.inputs[0]))])
        return Model(self.model.inputs[0], score_out)

    def get_generator_trained_model(self):
        """ Get the generator to reconstruct the input. Call it after training. """
        return Model(self.model.inputs[0], self.model.layers[1](self.model.inputs[0]))

    def get_discrinminator_trained_model(self):
        """ Get the discrinminator to reconstruct the input. Call it after training. """
        return Model(self.model.inputs[0], self.model.layers[2](self.model.layers[1](self.model.inputs[0])))

    def train(self, x_train, x_test, y_train, y_test, epochs=1):
        import scipy.ndimage as ndi
        x_train = ndi.zoom(x_train, (1, 32 / 28, 32 / 28), order=2)
        x_test = ndi.zoom(x_test, (1, 32 / 28, 32 / 28), order=2)
        x_train = np.expand_dims((x_train.astype('float32') - 127.5) / 127.5, axis=-1)
        x_test = np.expand_dims((x_train.astype('float32') - 127.5) / 255., axis=-1)
        self.model.add_loss(K.mean(self.g_loss))
        self.model.add_metric(self.g_loss, aggregation='mean', name="g_loss")
        self.model.add_loss(K.mean(self.d_loss))
        self.model.add_metric(self.d_loss, aggregation='mean', name="d_loss")

        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            # Train generator only
            self.model.layers[1].trainable = True
            self.model.layers[2].trainable = False
            self.model.compile('adam', loss_weights={'g_loss': 1, 'd_loss': 0})
            print('Training on Generator')
            self.model.fit(
                x_train,
                batch_size=64,
                steps_per_epoch=200,
                callbacks=[
                    LearningRateScheduler(
                        lr_scheduler(initial_lr=1e-3, decay_factor=0.75, step_size=10, min_lr=1e-5)
                    )
                ],
                initial_epoch=epoch
            )

            # Train discriminator only
            self.model.layers[1].trainable = False
            self.model.layers[2].trainable = True
            self.model.compile('adam', loss_weights={'g_loss': 0, 'd_loss': 1})
            print('Training on Discriminator')
            self.model.fit(
                x_train,
                batch_size=64,
                steps_per_epoch=200,
                callbacks=[
                    ModelCheckpoint(
                        './model_checkpoint/model_gloss_{g_loss:.4f}_dloss_{d_loss:.4f}.h5',
                        verbose=1
                    ),
                    LearningRateScheduler(
                        lr_scheduler(initial_lr=1e-3, decay_factor=0.75, step_size=10, min_lr=1e-5)
                    )
                ],
                initial_epoch=epoch
            )

if __name__ == '__main__':
    adae = ADAE(image_size=(32, 32, 1))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 8))
    test_filter = np.where((y_test == 8))
    x_train = x_train[train_filter]
    x_test = x_test[test_filter]
    adae.train(x_train, x_test, y_train, y_test, epochs=200)
