import sklearn.cluster
import sklearn.metrics
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# NOTE: In instances where the number of clusters is not known, should look into different heuristics for guessing. 

class Clustering():
    def __init__(self, n_clusters:int):

        self.n_clusters = n_clusters
        self.clusterer = None
        # It does not seem as though ScikitLearn clustering scales automatically, and distance-based clustering algorithms
        # are sensitive to differences in scale between features. 
        self.scaler = StandardScaler()
    
    @staticmethod
    def silhouette_score(X:np.ndarray, y_pred:np.ndarray, metric:str='euclidean'):
        return sklearn.metrics.silhouette_score(X, y_pred, metric=metric) 

    @staticmethod
    def rand_index(y_pred:np.ndarray, y_true:np.ndarray):
        '''Computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs 
        that are assigned in the same or different clusters in the predicted and true clusterings. This number is then adjusted
        according to the maximum and expected RI to fall between 0 and 1.'''
        return sklearn.metrics.adjusted_rand_score(y_true, y_pred) # This is a symmetric measure. 

    def fit(self, X:np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.clusterer.fit(X_scaled)

    def predict(self, X:np.ndarray):
        '''Returns an array of length n_samples containing the cluster labels.'''
        X_scaled = self.scaler.transform(X)
        return self.clusterer.predict(X_scaled)


class KMeansClustering(Clustering):

    def __init__(self, n_clusters:int):
        super().__init__(n_clusters)
        self.clusterer = sklearn.cluster.KMeans(n_clusters)


class SpectralClustering(Clustering):

    def __init__(self, n_clusters:int):
        super().__init__(n_clusters)
        self.clusterer = sklearn.cluster.SpectralClustering(n_clusters)

