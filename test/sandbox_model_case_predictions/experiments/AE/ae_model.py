import shutil
from os.path import join

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Model, regularizers


class Autoencoder(Model):
  def __init__(self, dir_output, dim_latent_non_categorical, dim_latent_categorical, dim_latent_count,
               dim_input_non_categorical, dim_input_categorical, dim_input_count):
    shutil.copy(__file__, join(dir_output, 'ae_model.py'))
    super(Autoencoder, self).__init__()


    ##### define input variables #####
    input_non_categorical = tf.keras.Input(shape=(dim_input_non_categorical,))
    input_categorical = tf.keras.Input(shape=(dim_input_categorical,))
    input_count = tf.keras.Input(shape=(dim_input_count,))
    input_revised_not_revised = tf.keras.Input(shape=(1,), dtype=tf.float32)

    ##### ensure the latent space is always lower dimensional than the input #####
    dim_latent_non_categorical = np.min([dim_latent_non_categorical, dim_input_non_categorical-1])
    dim_latent_categorical = np.min([dim_latent_categorical, dim_input_categorical-1])
    dim_latent_count = np.min([dim_latent_count, dim_input_count-1])




    ##### AE categorical #####
    encoder_categorical = tf.keras.Sequential([
      # tf.keras.layers.Dense(int(dim_input_categorical / 2), activation='tanh'),
      # tf.keras.layers.Dense(int(self.dim_latent * 2), activation='tanh'),
      # tf.keras.layers.BatchNormalization(),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(dim_latent_categorical, name='encoder_categorical')
    ])
    input_categorical_encoded = encoder_categorical(input_categorical)

    decoder_categorical = tf.keras.Sequential([
      # tf.keras.layers.Dense(int(dim_input_categorical / 2), activation='tanh'),
      # tf.keras.layers.BatchNormalization(),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(dim_input_categorical, activation='sigmoid', name='decoder_categorical'),
    ], name='reconstruction_categorical')
    reconstructed_input_categorical = decoder_categorical(input_categorical_encoded)

    decoder_categorical_binary = tf.keras.Sequential([
      tf.keras.layers.Lambda(lambda x: K.round(x))
    ])
    reconstructed_input_categorical_binary = decoder_categorical_binary(reconstructed_input_categorical)




    ##### AE none categorical #####
    encoder_non_categorical = tf.keras.Sequential([
      # tf.keras.layers.Dense(dim_input_non_categorical, activation='relu'),
      # tf.keras.layers.Dense(int(dim_input_non_categorical / 2), activation='relu'),
      # tf.keras.layers.BatchNormalization(),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(dim_latent_non_categorical, name='encoder_non_categorical')
    ])
    input_non_categorical_encoded = encoder_non_categorical(input_non_categorical)

    decoder_non_categorical = tf.keras.Sequential([
      # tf.keras.layers.Dense(int(dim_input_non_categorical / 2), activation='relu'),
      # tf.keras.layers.BatchNormalization(),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(dim_input_non_categorical, activation='sigmoid', name='decoder_non_categorical')
    ], name='reconstruction_non_categorical')
    reconstructed_input_non_categorical = decoder_non_categorical(input_non_categorical_encoded)




    ##### AE count #####
    encoder_count = tf.keras.Sequential([
      # tf.keras.layers.Dense(dim_input_non_categorical, activation='relu'),
      # tf.keras.layers.Dense(int(dim_input_non_categorical / 2), activation='relu'),
      # tf.keras.layers.BatchNormalization(),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(dim_latent_count, name='encoder_count', kernel_initializer=tf.keras.initializers.constant(value=0))
    ])
    input_count_encoded = encoder_count(input_count)

    decoder_count = tf.keras.Sequential([
      # tf.keras.layers.Dense(int(dim_input_non_categorical / 2), activation='tanh'),
      # tf.keras.layers.BatchNormalization(),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(dim_input_count, name='decoder_count', kernel_initializer=tf.keras.initializers.constant(value=0),
                            activation=tf.keras.activations.exponential)
    ], name='reconstruction_count')
    reconstructed_input_count = decoder_count(input_count_encoded)

    decoder_count_binary = tf.keras.Sequential([
      tf.keras.layers.Lambda(lambda x: K.round(x))
    ])
    reconstructed_input_count_binary = decoder_count_binary(reconstructed_input_count)



    ##### combined latent space #####
    latent_space = tf.keras.layers.Concatenate(name='latent_space')([input_categorical_encoded, input_non_categorical_encoded, input_count_encoded])



    ##### predict revision on latent space #####
    model_predict_revision = tf.keras.Sequential([
      # tf.keras.layers.Dense(dim_latent-1, activation='elu'),
      # tf.keras.layers.Dropout(rate=0.5),
      tf.keras.layers.Dense(1, name='prediction_revision', activation='sigmoid')
    ], name='predictions_revisions')
    prediction_revision = model_predict_revision(latent_space)



    self.model_train = tf.keras.models.Model(inputs=[input_categorical, input_non_categorical, input_count, input_revised_not_revised],
                                             outputs=[reconstructed_input_categorical, reconstructed_input_non_categorical, reconstructed_input_count, prediction_revision])
    # self.model_train.compile(optimizer='adam', loss=[tf.keras.losses.binary_crossentropy, tf.keras.losses.mse, tf.keras.losses.binary_crossentropy])

    # def poisson(y_true, y_pred):
    #   return tf.reduce_mean(tf.keras.losses.poisson(y_true, y_pred))

    self.model_train.compile(optimizer='adam',
                             loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.mse, tf.keras.losses.Poisson(), tf.keras.losses.BinaryCrossentropy()],
                             loss_weights=[1,1,1,3])


    self.model_latent_space = tf.keras.Model([input_categorical, input_non_categorical, input_count], latent_space)
    self.model_reconstruction_categorical = tf.keras.Model(input_categorical, reconstructed_input_categorical_binary)
    self.model_reconstruction_non_categorical = tf.keras.Model(input_non_categorical, reconstructed_input_non_categorical)
    self.model_reconstruction_count = tf.keras.Model(input_count, reconstructed_input_count_binary)
    self.model_predict_revision = tf.keras.Model([input_categorical, input_non_categorical, input_count], prediction_revision)

