"""The simplest possible GAN for estimating a source site distribution."""

from time import time

import openmc
import tensorflow as tf


def make_generator(n_input, n_hidden, n_output):
    """Create the generator network.

    Parameters
    ----------
    n_input : int
        Number of inputs in random, latent input vector
    n_hidden : int
        Number of hidden units in network architecture
    n_output : int
        Number of outputs in "fake" sampled data
        (e.g., number of source sites in 1, 2, or 3D)

    Returns
    -------
    model : tensorflow.keras.Sequential
        A Tensorflow 2.0 model of fully-connected and batchnorm
        layers with ReLU activations

    """

    model = tf.keras.Sequential()

    # First layer
    model.add(tf.keras.layers.Dense(units=n_hidden, input_shape=(n_input,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))

    # Second layer
    model.add(tf.keras.layers.Dense(units=n_hidden, input_shape=(n_hidden,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))

    # Third layer
    model.add(tf.keras.layers.Dense(units=n_output, input_shape=(n_hidden,)))

    return model


def make_discriminator(n_input, n_hidden):
    """Create the discriminator network.

    Parameters
    ----------
    n_input : int
        Number of inputs in source site vector
        (e.g. number of source sites in 1, 2 or 3D)
    n_hidden : int
        Number of hidden units in network architecture

    Returns
    -------
    model : tensorflow.keras.Sequential
        A Tensorflow 2.0 model of fully-connected and batchnorm
        layers with ReLU activations and ending with a sigmoid

    """

    model = tf.keras.Sequential()

    # First layer
    model.add(tf.keras.layers.Dense(units=n_hidden, input_shape=(n_input,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))

    # Second layer
    model.add(tf.keras.layers.Dense(units=n_hidden, input_shape=(n_hidden,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))

    # Third layer
    model.add(tf.keras.layers.Dense(units=1, input_shape=(n_hidden,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model

def discriminator_loss(real_output, fake_output):
    """
    """

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_disc_output, gen_output, target, lambd=100):
    """
    """

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gen_loss = loss_object(tf.ones_like(fake_disc_output), fake_disc_output)

    # Mean Absolute Error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_loss = gen_loss + (lambd * l1_loss)
    return total_loss


if __name__ == "__main__":

    # Define parameters
    n_input = 100     # Length of random, latent vector input to generator
    n_hidden = 1000   # Number of hidden units in generator/discriminator
    n_sites = 100     # Number of source sites output by GAN

    n_epochs = 10     # Number of epochs for training
    n_batches = 5     # Number of minibatches per epoch
    batch_size = 5    # Number of source site vectors per minibatch

    # Create generator and discriminator networks
    generator = make_generator(n_input, n_hidden, n_sites * 3)
    discriminator = make_discriminator(n_sites, n_hidden)

    # Initialize Adam optimizers for both networks with LR=1e-4
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # TODO: Implement loss function for generator
    # TODO: Implement loss function for discriminator

    # Sample from uniform distribution for input to generator
    noise = tf.random.normal([1, n_sites])

    # TODO: Prep dataset
    # Open OpenMC statepoint
    sp = openmc.StatePoint("statepoint.10.h5")

    # TODO: Add data augmentation

    hmm = generator(noise)

    # TODO: Setup checkpointing

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

    @tf.function
    def train_step(input, target):
        """
        """

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            gen_fake_output = generator(input, training=True)

            disc_real_output = discriminator([input, target], training=True)
            disc_fake_output = discriminator([input, gen_fake_output], training=True)

            gen_loss = generator_loss(disc_fake_output, gen_fake_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

        generator_gradients = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):
            t0 = time()

            for i_batch in range(n_minibatches):

                train_step(input_image, target)


            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print (f"Time taken for epoch {epoch+1} is {time()-t0} sec")