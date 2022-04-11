import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import InputLayer, Dense
from keras.models import Model

initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2.0))


class Actor(Model):

    def __init__(self, input_dim, output_dim, hidden_units):
        super(Actor, self).__init__()

        self.input_layer = InputLayer(input_shape=input_dim)
        self.hidden1 = Dense(hidden_units[0], activation='relu', kernel_initializer=initializer)
        self.hidden2 = Dense(hidden_units[1], activation='relu', kernel_initializer=initializer)
        self.mu = Dense(output_dim, activation='tanh')
        self.log_sigma = Dense(output_dim, activation='tanh')

    @tf.function
    def call(self, state):
        x = self.input_layer(state)
        x = self.hidden1(x)
        x = self.hidden2(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)

        sigma = tf.exp(log_sigma)
        distribution = tfp.distributions.Normal(mu, sigma)
        action = distribution.sample()

        return action, distribution


class Critic(Model):

    def __init__(self, input_dim, hidden_units):
        super(Critic, self).__init__()

        self.input_layer = InputLayer(input_shape=input_dim)
        self.hidden1 = Dense(hidden_units[0], activation='relu', kernel_initializer=initializer)
        self.hidden2 = Dense(hidden_units[1], activation='relu', kernel_initializer=initializer)
        self.value = Dense(1, activation=None)

    @tf.function
    def call(self, state):
        x = self.input_layer(state)
        x = self.hidden1(x)
        x = self.hidden2(x)
        value = self.value(x)

        return value
