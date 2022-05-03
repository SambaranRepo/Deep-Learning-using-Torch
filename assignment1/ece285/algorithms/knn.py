"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
        labels = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]): 
            neighbours = []
            neighbours = self._y_train[np.argsort(distance[i,:])][:k]
            labels[i] = np.argmax(np.bincount(neighbours))
        
        return labels

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        # TODO: implement me
        N_train = self._x_train.shape[0]
        N_test = x_test.shape[0]
        dist = np.zeros((N_test, N_train), dtype = np.float32)
        
        for i in range(N_test): 
            dist[i] = np.sum(np.square(x_test[i] - self._x_train), axis = 1)
        
        return dist

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        N_train = self._x_train.shape[0]
        N_test = x_test.shape[0]
        dist = np.zeros((N_test, N_train), dtype = np.float32)
        
        for i in range(N_test): 
            for j in range(N_train): 
                dist[i,j] = np.sum(np.square(x_test[i] - self._x_train[j]))
        
        return dist
