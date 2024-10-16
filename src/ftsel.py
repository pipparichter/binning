'''Implementation of various algorithms for feature selection.'''

import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.feature_selection import f_classif
from sklearn.neighbors import kneighbors_graph 
import time
import sklearn
from tqdm import tqdm 
from typing import NoReturn
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances
# from scipy.stats import f_oneway

# Need to think about a good, consistent interface for these... 
# Probably will stick to working with Numpy arrays instead of Tensors. 


class UnivariateFilter():
    def __init__(self, low_memory:bool=True):

        self.order = None # This will be populated after fitting. It stores the features in order of decreasing importance. 
        self.low_memory = low_memory


    def fit(self, embeddings:np.ndarray, labels:np.ndarray=None, **kwargs):
        pass 

    def transform(self, embeddings:np.ndarray, n_features:int=3):
        '''Use the stored feature order to select the most relevant features.'''
        assert embeddings.shape[-1] == len(self.order), f'UnivariateFilter: The number of scores ({len(self.order)}) does not match the input dimensions ({embeddings.shape[-1]}).'
        # Always assume order is from most to least important features!
        idxs = self.order[:n_features]
        return embeddings[:, idxs]

    def distance_matrix(self, X:np.ndarray, metric:str='euclidean') -> np.ndarray:
        
        n = len(X) # Get the number of samples. 
        memory = (np.dtype(np.float16).itemsize * n * n) * 1e-9 if self.low_memory else (np.dtype(X.dtype).itemsize * n * n) * 1e-9
        print(f'UnivariateFilter.distance_matrix: Predicted to use {int(memory)} GB of memory.')
        if self.low_memory:
            D = []
            #working_memory = sklearn.get_config()['working_memory'] / 1000 # This is in units MiB, so convert to GB. 
            working_memory = 100
            n_chunks = (memory // (working_memory * 1e-3)) + 1 # Approximate the number of chunks required. 
            chunks = pairwise_distances_chunked(X, metric=metric, reduce_func=lambda x, i : x.astype(np.float16))
            for chunk in tqdm(chunks, total=int(n_chunks), desc=f'UnivariateFilter: Computing the distance matrix with metric {metric} in low-memory mode...'):
                D.append(chunk)
            D = np.array(D)
        else:
            D = pairwise_distances(X, metric=metric)
        return D



class SUD(UnivariateFilter):
    '''Implementation of Sequential backward selection method for Unsupervised Data, as described
    here: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=da15debfe12f94275059ef375740240053fad4bc. 
    This is an unsupervised, information-based approach to feature selection.'''
    # NOTE: What distance metric should I use? Euclidean? Cosine similarity?
    # Cosine similarity measures the angle between two vectors, not the magnitude. Euclidean accounts for magnitude. 
    # Cosine similarity is the normalized dot product of two vectors. 

    def __init__(self, metric='euclidean'):

        super().__init__()
        # self.D = None # For storing the distance matrix, which I think would be intensive to keep re-computing. 
        self.alpha = None
        self.metric = metric 
        self.dist_func = euclidean_distances if (metric == 'euclidean') else cosine_similarity

    
    def similarity(self, i:int, j:int):
        return np.exp(-self.D[i, j] * self.alpha)

    def entropy(self, X:np.ndarray):
        '''Compute the entropy metric, as described in the paper.'''
        E = 0
        for i in range(len(X)):
            for j in range(len(X)):
                S = self.similarity(i, j)
                E += (S * np.log(S) + (1 - S) * np.log(1 - S))
        return E


    def fit(self, embeddings:np.ndarray):
        '''Compute the order of features in order of importance using the SUD algorithm.'''

        embeddings = embeddings.astype(np.float16) if self.low_memory else embeddings
        print(embeddings.dtype)

        n = len(embeddings) # The number of embeddings. 
        d = embeddings.shape[-1] # The original dimension of the embeddings. 

        # D = np.zeros((n, n)) # Initialize the distance matrix. 
        self.D = self.distance_matrix(embeddings, metric=self.metric) # Compute the distance matrix. 
        self.alpha = -np.log(0.5) / np.mean(self.D) # Use the alpha value from the paper. 

        order = [] # Store the order in which features are removed. 
        X = embeddings.copy()
        for _ in tqdm(range(d), desc='SUD.fit: Computing entropy scores for each feature...'):
            min_E, min_i = np.inf, None
            for i in range(embeddings.shape[-1]):
                Xi = np.delete(X, i, axis=1)
                E = self.entropy(Xi)
                if E < min_E:
                    min_E = E
                    min_i = i
            X = np.delete(X, min_i, axis=1) # REmove the feature for which E is minimized. 
            order.append(min_i)
    
        # Order list is currently in order or worst to best, which we want to reverse. 
        self.order = np.array(order[::-1])
        


class LaplacianScore(UnivariateFilter):
    '''A spectrum-based univariate filter method for feature selection. According to the review by Solorio-Fernandez, 
    this is one of the most referenced and relevant method in this category. It is described in this paper:
    https://proceedings.neurips.cc/paper_files/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf'''

    def __init__(self, metric:str='euclidean', t:float=1.0, k:int=50):
        ''' Initialize a class for feature selection using the Laplacian score.
        
        :param metric: One of the metrics listed here https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
        ''' 


        self.D = None # For storing the distance matrix, which I think would be intensive to keep re-computing. 
        self.metric = metric # For cosine similarity, use 'cosine'.
        # self.dist_func = euclidean_distances if (metric == 'euclidean') else cosine_similarity
        self.t = t # Constant to use when computing the similarity score. 
        self.k = k # The k for constructing the k-nearest neighbors graph. 
        # self.S = None # For storing the weight matrix. 

    def similarity_matrix(self, X:np.ndarray, k:int=None): # NOTE: What to use as the default for this?
        '''Create a k-nearest neighbors graph using the distance matrix. An edge is added between nodes i and j if 
        xi is among the k nearest neighbors of xj OR xj is among the k nearest neighbors of xi.
        
        :param X: An array of shape (n_samples, n_features).
        :param k: The number of nearest neighbors for drawing edges.         
        '''
        # NOTE: Not sure whether I should convert the CSR representation back into a dense matrix. Indexing the sparse matrix is slower, 
        # but the dense matrix will liekly take up more memory (depending on the value of k). 
        # Setting mode to connectivity returns an adjacency matrix. 
        A = kneighbors_graph(X, n_neighbors=k, metric=self.metric, mode='connectivity').toarray() # This returns a sparse matrix in CSR (compressed sparse row) format. 
        print('LaplacianScore.similarity_matrix: Computed k-nearest neighbor graph using the embeddings.')
        W = kneighbors_graph(X, n_neighbors=k, metric='euclidean', mode='distance').toarray() # Get the distance matrix. 
        print('LaplacianScore.similarity_matrix: Computed matrix Euclidean distance between points.')
        G = W * A # Get the weighted graph with the Hadamard product. 

        # Compute the weight matrix S of the graph. 
        n = len(X) # The number of samples in the dataset. 
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = 0 if (G[i, j] == 0) else np.exp(-G[i, j] / self.t)
        
        print('LaplacianScore.fit: Computed matrix of similarity scores.')

        return S
    

    def fit(self, embeddings:np.ndarray):

        n, d = embeddings.shape # n is the number of samples, d is the number of features. 

        S = self.similarity_matrix(embeddings, k=self.k)
        D = np.diagonal(S @ np.ones(n).T)
        L = D - S
        ones = np.ones(n).T

        scores = []
        for r in tqdm(range(d), desc='LaplacianScore.fit: Computing Laplacian scores...'): # Iterate over the features. 
            fr = embeddings[:, r]
            fr = fr - (fr.T @ D @ ones) / (ones.T @ D @ ones) @ ones # Adjust the vector, kind of like removing the mean. 
            assert ones.T @ D @ ones == np.sum(D), 'LaplacianScore.fit: I think these should be the same.'
            Lr = (fr.T @ L @ fr) / (fr.T @ D @ fr)
            scores.append(Lr)
        scores = np.ndarray()
        self.order = np.argsort(scores) # Increasing order, so order is from best to worst features. 




class ANOVA(UnivariateFilter):
    '''Implementation of ANOVA for feature selection. This is a supervised method, which we can use as a benchmark 
    for the unsupervised approaches.''' 
    
    def __init__(self):
        super().__init__()


    def fit(self, embeddings:np.ndarray, labels:np.ndarray):
        '''Selects the most important features using a one-way ANOVA test. I think the idea here is just to 
        apply ANOVA to each feature, independently, and select the features which "score" the best, i.e. 

        ''' 
        # A higher F-ratio indicates that the treatment variables are significant, i.e. within-group variance 
        # is higher than the background. 
        d = embeddings.shape[-1] # The original dimension of the embeddings. 

        f_statistics, p_values = f_classif(embeddings, labels)
        order = np.argsort(f_statistics) # Order will be from least important to most important. 
        self.order = order[::-1] # Make the order most to least important for consistency. 

