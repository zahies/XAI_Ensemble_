from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential

import argparse
import os
global original_dim
global intermediate_dim

original_dim = None
input_shape = (original_dim,)
intermediate_dim = None
batch_size = 128
latent_dim = 64
epochs = 80
epsilon_std = 1.0



class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def define_fit_VAE(x_train, x_test, data_path):
    # network parameters
    original_dim = x_train.shape[1]-1
    intermediate_dim = int(original_dim / 2)

    # VAE Architecture
    # * original_dim - Original Input Dimension
    # * intermediate_dim - Hidden Layer Dimension
    # * latent_dim - Latent/Embedding Dimension

    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    # Encode
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                       shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred = decoder(z)
    return x, eps, z_mu, x_pred


def fit_VAE(x_train, x_test, data_path):
    x, eps, z_mu, x_pred = define_fit_VAE(x_train, x_test, data_path)
    vae = Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer='adam', loss=nll)

    checkpoint = ModelCheckpoint(data_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    hist = vae.fit(x_train, x_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   callbacks=callbacks_list,
                   validation_data=(x_test, x_test))

    return vae





def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)



