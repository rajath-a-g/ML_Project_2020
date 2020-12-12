import os

import keras.backend as bk
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Flatten, Reshape, Dense, Lambda, Conv2DTranspose, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 2


class VAE():
    def __init__(self):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        # 28,28,1
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        optimizer = Adam(learning_rate=0.0005)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        VAE_inp = Input(shape=self.img_shape, name='VAE_input')
        encoder_output = self.encoder(VAE_inp)
        decoder_output = self.decoder(encoder_output)
        self.vae_model = Model(VAE_inp, decoder_output, name='VAE')
        self.vae_model.summary()
        self.vae_model.compile(optimizer=optimizer, loss=self.loss_func())

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = bk.random_normal(shape=bk.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + bk.exp(log_variance / 2) * epsilon
        return random_sample

    def loss_func(self):
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1000
            reconstruction_loss = bk.mean(bk.square(y_true - y_predict), axis=[1, 2, 3])
            return reconstruction_loss_factor * reconstruction_loss

        def vae_kl_loss(y_true, y_predict):
            kl_loss = -0.5 * bk.sum(1.0 + y_predict - bk.square(y_true) - bk.exp(y_predict), axis=1)
            return kl_loss

        # def vae_kl_loss_metric(y_true, y_predict):
        #   kl_loss = -0.5 * bk.sum(1.0 + self.log - bk.square(self.mu) - bk.exp(self.log), axis=1)
        #   return kl_loss

        def vae_loss(y_true, y_predict):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
            kl_loss = vae_kl_loss(y_true, y_predict)
            loss = reconstruction_loss + kl_loss
            return loss

        return vae_loss

    def build_encoder(self):
        inp = Input(shape=self.img_shape, name="encoder_input")
        conv1 = Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=1, name="encoder_conv_1")(inp)
        norm1 = BatchNormalization(name="encoder_norm_1")(conv1)
        leakyrelu1 = LeakyReLU(name="encoder_leakyrelu_1")(norm1)

        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=1, name="encoder_conv_2")(leakyrelu1)
        norm2 = BatchNormalization(name="encoder_norm_2")(conv2)
        leakyrelu2 = LeakyReLU(name="encoder_leakyrelu_2")(norm2)

        conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="encoder_conv_3")(leakyrelu2)
        norm3 = BatchNormalization(name="encoder_norm_3")(conv3)
        leakyrelu3 = LeakyReLU(name="encoder_leakyrelu_3")(norm3)

        conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="encoder_conv_4")(leakyrelu3)
        norm4 = BatchNormalization(name="encoder_norm_4")(conv4)
        leakyrelu4 = LeakyReLU(name="encoder_leakyrelu_4")(norm4)

        conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", strides=1, name="encoder_conv_5")(leakyrelu4)
        norm5 = BatchNormalization(name="encoder_norm_5")(conv5)
        leakyrelu5 = LeakyReLU(name="encoder_leakyrelu_5")(norm5)

        self.shape_before_flatten = bk.int_shape(leakyrelu5)[1:]
        flatten = Flatten(name='encoder_flatten')(leakyrelu5)

        self.mu = Dense(units=self.latent_dim, name="encoder_mu")(flatten)
        self.log = Dense(units=self.latent_dim, name="encoder_log_variance")(flatten)

        # model = Model(inp, (mu, log), name='VAE_encoder')
        output = Lambda(self.sampling, name='encoder_output')([self.mu, self.log])

        model = Model(inp, output, name='VAE_encoder')
        model.summary()
        return model

    def build_decoder(self):
        inp = Input(shape=self.latent_dim, name="decoder_input")
        dense1 = Dense(units=np.prod(self.shape_before_flatten), name="decoder_dense_1")(inp)
        resh = Reshape(self.shape_before_flatten)(dense1)

        conv_trans1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=1,
                                      name="decoder_conv_tran_1")(resh)
        norm_layer1 = BatchNormalization(name="decoder_norm_1")(conv_trans1)
        leakyrelu1 = LeakyReLU(name="decoder_leakyrelu_1")(norm_layer1)

        conv_trans2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2,
                                      name="decoder_conv_tran_2")(leakyrelu1)
        norm_layer2 = BatchNormalization(name="decoder_norm_2")(conv_trans2)
        leakyrelu2 = LeakyReLU(name="decoder_leakyrelu_2")(norm_layer2)

        conv_trans3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2,
                                      name="decoder_conv_tran_3")(leakyrelu2)
        norm_layer3 = BatchNormalization(name="decoder_norm_3")(conv_trans3)
        leakyrelu3 = LeakyReLU(name="decoder_leakyrelu_3")(norm_layer3)

        conv_trans4 = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding="same", strides=1,
                                      name="decoder_conv_tran_4")(leakyrelu3)
        output = LeakyReLU(name="decoder_leakyrelu_4")(conv_trans4)

        model = Model(inp, output, name='VAE_decoder')
        model.summary()
        return model

    def train(self, epochs, x_train, batch_size=64):
        his = self.vae_model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                 validation_split=0.2)
        self.encoder.save("VAE_encoder.h5")
        self.decoder.save("VAE_decoder.h5")
        self.vae_model.save("VAE.h5")
        return his

    def predict(self, x_test):
        pred = self.vae_model.predict(x_test)
        self.sample_images(pred)
        return pred

    def sample_images(self, pred):
        r, c = 5, 5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(pred[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/VAE/vae.png")
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("./images/VAE"):
        os.makedirs("./images/VAE")
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_total = np.concatenate((X_train, X_test), axis=0)
    if not os.path.exists("VAE.h5"):
        vae = VAE()
        his = vae.train(x_train=X_total, epochs=50, batch_size=150)
        print('loss: ' + str(np.mean(his.history['loss'])) + ', val_loss:' + str(np.mean(his.history['val_loss'])))
        plt.figure()
        plt.plot(his.epoch, his.history['loss'], label='loss')
        plt.plot(his.epoch, his.history['val_loss'], label='val_loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("images/VAE/vae_loss.jpg")
    else:
        encoder = load_model("VAE_encoder.h5", compile=False)
        decoder = load_model("VAE_decoder.h5", compile=False)
        vae = load_model("VAE.h5", compile=False)
    vae.predict(X_train[:25])
