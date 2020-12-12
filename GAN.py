import os

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        # plot_model(self.discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        # plot_model(self.generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential(name='discriminator')
        model.add(Flatten(input_shape=self.img_shape))
        # model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.4))

        model.add(Dense(256))
        # model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_generator(self):
        model = Sequential(name='generator')

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Reshape((7, 7, 128)))

        model.add(Dense(512))
        # model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # .add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, t_data, batch_size=64, sample_interval=50):
        res_g_loss = []
        t_data = t_data / 127.5 - 1.
        t_data = np.expand_dims(t_data, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, t_data.shape[0], batch_size)
            imgs = t_data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            res_g_loss.append(g_loss)
            print("%d [d_loss_real: %.4f, d_loss_fake: %.4f] [g_loss: %.4f]" % (epoch, d_loss[0], d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        return res_g_loss

    def sample_images(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_total = np.concatenate((X_train, X_test), axis=0)
    gan = GAN()

    g_loss = gan.train(epochs=30000, batch_size=200, sample_interval=100, t_data=X_total)
    print(np.mean(g_loss))
    plt.figure()
    plt.plot(g_loss)
    plt.xlabel("epochs")
    plt.ylabel("g_loss")
    plt.savefig("images/g_loss.jpg")
