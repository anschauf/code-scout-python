import numpy as np
import torch
from torch.utils.data import Sampler


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.batch_size = batch_size
        self.class_vector = class_vector
        self.ind_negatives = np.where(class_vector.detach().numpy() == -1)[0]
        self.ind_positives = np.where(class_vector.detach().numpy() == 1)[0]

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        # s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        # X = torch.randn(self.class_vector.size(0), 2).numpy()
        # y = self.class_vector.numpy()
        # s.get_n_splits(X, y)
        #
        # train_index, test_index = next(s.split(X, y))
        # return np.hstack([train_index, test_index])
        ind_negative = np.random.choice(self.ind_negatives, size=(int(self.batch_size/2),), replace=False)
        ind_positive = np.random.choice(self.ind_positives, size=(int(self.batch_size/2),), replace=False)
        ind_shuffle = np.random.choice(range(self.batch_size), replace=False, size=(self.batch_size, ))
        return np.hstack([ind_positive, ind_negative])[ind_shuffle]

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)