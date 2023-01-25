import keras
import numpy as np


class BalancedGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size, features_categorical, features_non_categorical, features_count, y_train):
        'Initialization'
        self.batch_size = batch_size
        self.y_train = y_train
        self.ind_train_positive = np.where(y_train == 1)[0]
        self.ind_train_negative = np.where(y_train == 0)[0]
        self.features_categorical = features_categorical
        self.features_non_categorical = features_non_categorical
        self.features_count = features_count

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.features_categorical)/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        ind_train_positive_batch = np.random.choice(self.ind_train_positive, size=(int(self.batch_size / 2),))
        ind_train_negative_batch = np.random.choice(self.ind_train_negative, size=(int(self.batch_size / 2),))
        ind_batch = np.concatenate([ind_train_positive_batch, ind_train_negative_batch])
        data = [np.asarray(self.features_categorical[ind_batch]),
                np.asarray(self.features_non_categorical[ind_batch]),
                np.asarray(self.features_count[ind_batch]),
                np.asarray(self.y_train[ind_batch])]

        return data, data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

