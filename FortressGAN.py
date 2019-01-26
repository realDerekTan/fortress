# powered by Issac AI
# written by Lleyton Ariton

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Dropout, GaussianNoise, Conv2DTranspose
from keras.layers import ReLU, LeakyReLU, Flatten, BatchNormalization, UpSampling2D, Reshape
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import Flatten, Concatenate, concatenate
from keras.layers import GRU, LSTM, Bidirectional, Embedding, Input


class IsaacGAN:
    """
        I do not claim to own any of the IsaacGAN code. Though it was modified, the essence of the code is still credited to Rowel Atienza.
        His original code that IsaacGAN was built upon can be found here: https://github.com/roatienza/Deep-Learning-Experiments
        """

    def __init__(self, data_filepath, output_data_filepath, epochs=1000, batchsize=256, checkpoint=50):
        def preprocess_data(file_path):
            data = []
            for person in os.listdir(file_path):
                for image in os.listdir(file_path + '/' + person):
                    path = file_path + '/' + person + '/' + image
                    data.append(img_to_array(Image.open(path).resize((28, 28)).convert('L')))
            data = np.array(data)
            return data

        self.data = preprocess_data(data_filepath)

        def create_discriminator():
            discriminator_input = Input(shape=(28, 28, 1))
            discriminator = Conv2D(65, 5, strides=(2, 2), padding='same')(discriminator_input)
            discriminator = LeakyReLU()(discriminator)
            discriminator = Dropout(0.4)(discriminator)
            discriminator = Conv2D(256, 5, strides=(2, 2), padding='same')(discriminator)
            discriminator = LeakyReLU()(discriminator)
            discriminator = Dropout(0.4)(discriminator)
            discriminator = Conv2D(512, 5, strides=(2, 2), padding='same')(discriminator)
            discriminator = LeakyReLU()(discriminator)
            discriminator = Dropout(0.4)(discriminator)
            discriminator = Flatten()(discriminator)
            discriminator_output = Dense(1, activation='sigmoid')(discriminator)
            discriminator = Model(discriminator_input, discriminator_output)
            discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            discriminator.summary()
            return discriminator

        def create_generator():
            generator_input = Input(shape=(100,))
            generator = Dense(12544)(generator_input)
            generator = BatchNormalization(momentum=0.9)(generator)
            generator = ReLU()(generator)
            generator = Reshape((7, 7, 256))(generator)
            generator = Dropout(0.4)(generator)
            generator = UpSampling2D()(generator)
            generator = Conv2DTranspose(128, 5, padding='same')(generator)
            generator = BatchNormalization(momentum=0.9)(generator)
            generator = ReLU()(generator)
            generator = UpSampling2D()(generator)
            generator = Conv2DTranspose(64, 5, padding='same')(generator)
            generator = BatchNormalization(momentum=0.9)(generator)
            generator = ReLU()(generator)
            generator = Conv2DTranspose(32, 5, padding='same')(generator)
            generator = BatchNormalization(momentum=0.9)(generator)
            generator = ReLU()(generator)
            generator_output = Conv2DTranspose(1, 5, padding='same', activation='sigmoid')(generator)
            generator = Model(generator_input, generator_output)
            generator.summary()
            return generator

        self.create_generator = create_generator()
        self.create_discriminator = create_discriminator()

        def combine():
            adversarial = Sequential()
            adversarial.add(self.create_generator)
            adversarial.add(self.create_discriminator)
            adversarial.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
            return adversarial

        self.combine = combine()

        def train_isaac_gan(x_train, generator_model, discriminator_model, adversarial_model, train_steps=2000,
                            batch_size=256,
                            save_interval=0):

            def plot_images(x_train, generator_model, save2file=False, fake=True, samples=16, noise=None, step=0):
                filename = 'isaac_gan_test.png'
                if fake:
                    if noise is None:
                        noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
                    else:
                        filename = output_data_filepath + '/' + 'isaac_gan_test_%d.png' % step
                    images = generator_model.predict(noise)
                else:
                    i = np.random.randint(0, x_train.shape[0], samples)
                    images = x_train[i, :, :, :]

                plt.figure(figsize=(10, 10))
                for i in range(images.shape[0]):
                    plt.subplot(4, 4, i + 1)
                    image_ = images[i, :, :, :]
                    image_ = np.reshape(image_, [28, 28])
                    plt.imshow(image_, cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                if save2file:
                    plt.savefig(filename)
                    plt.close('all')
                else:
                    plt.show()

            noise_input = None
            if save_interval > 0:
                noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
            for i in range(train_steps):
                images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                images_fake = generator_model.predict(noise)
                x = np.concatenate((images_train, images_fake))
                y = np.ones([2 * batch_size, 1])
                y[batch_size:, :] = 0
                d_loss = discriminator_model.train_on_batch(x, y)

                y = np.ones([batch_size, 1])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                a_loss = adversarial_model.train_on_batch(noise, y)
                log_mesg = "%d: Discriminator loss: %f, acc: %f -----" % (i, d_loss[0], d_loss[1])
                log_mesg = "%s  Adversarial Model loss: %f, acc: %f" % (log_mesg, a_loss[0], a_loss[1])
                print(log_mesg)
                if save_interval > 0:
                    if (i + 1) % save_interval == 0:
                        plot_images(x_train, generator_model, save2file=True, samples=noise_input.shape[0],
                                    noise=noise_input, step=(i + 1))

        self.train_isaac_gan = train_isaac_gan(self.data, self.create_generator, self.create_discriminator, self.combine, train_steps=epochs, batch_size=batchsize, save_interval=checkpoint)
