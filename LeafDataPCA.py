#!/usr/local/bin/python

import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from plotly.graph_objs import *

class App():
    def __init__(self, filename):
        print('\nFilename is: "%s"' % filename)
        
        # Loads the data
        self._leaf = pd.read_csv(filename, dtype=np.float64, header=None)
       
        # Spliting the leaf data into training data and class label.
        self._X = self._leaf.ix[:,2:15].values
        self._y = self._leaf.ix[:,0].values

        # Variable used for Normalization
        self._scaledSamples = None

        # Mean vector variable
        self._meanVector = None

        # Covariance Matrix variable
        self._covarianceMatrix = None

        # EigenVector and EigenValue
        self._eigenVectors = None
        self._eigenValues = None
        self._eigPairs = []

        # Variance explanation variable
        self._varExp = []
        
        
    def computeResults(self):
        """
        Computes covarinace matrix, eigen vectors and values. At the end it shows principal components. 
        """
        
        """
	Normalize the data by transforming it onto a unit scale (mean=0 and variance=1).
	Performing PCA on unnormalized data will lead to dependence on variable with high variance. 
	"""
        self._scaledSamples = StandardScaler().fit_transform(self._X)

        # Covariance matrix -> Covariance matrix represents the covariance between different features in dataset.
        # Process of centering the data.
        self._meanVector = np.mean(self._scaledSamples, axis=0)
        self._covarianceMatrix = (self._scaledSamples - self._meanVector).T.dot((self._scaledSamples - self._meanVector)) / (self._scaledSamples.shape[0]-1)
        
        """
	Implementing a EigenDecomposition on covariance matrix gives EigenValues and EigenVectors.
	Sum of the EigenValues gives the variance in the dataset.
	"""
        self._eigenValues, self._eigenVectors = np.linalg.eig(self._covarianceMatrix)
        
        print("\nComputed EigenVectors are: \n%s" %self._eigenVectors)
        print("\nComputed EigenValues are: \n%s" %self._eigenValues)
        
        # Make a list of (Eigenvalue, EigenVector) tuples.
        for i in range(len(self._eigenValues)):
            self._eigPairs.append((np.abs(self._eigenValues[i]), self._eigenVectors[:,i]))
   
        # Sort the (Eigenvalue, EigenVector) tuples from high to low.
        self._eigPairs.sort()
        self._eigPairs.reverse()
        
        # Calculating explained variance helps in selection of principal components.
        tot = sum(self._eigenValues)
        for i in sorted(self._eigenValues, reverse=True):
            self._varExp.append((i / tot)*100)
        cumVarExp = np.cumsum(self._varExp)

        print("\nFinal Cumulative Variance is: \n%s" %cumVarExp)
    
        print "\nFirst principal component: \n", self._eigPairs[0][1].reshape(14,1)
        # Flipping sign for components 2nd, 3rd and 4th.
        print "\nSecond principal component: \n", -(self._eigPairs[1][1].reshape(14,1))
        print "\nThird principal component: \n", -(self._eigPairs[2][1].reshape(14,1))
        print "\nFourth principal component: \n", -(self._eigPairs[3][1].reshape(14,1))
        print "\nFifth principal component: \n", -(self._eigPairs[4][1].reshape(14,1))
        
        self._verifyCalculation()
    
    def _verifyCalculation(self):
        """
        Inbuilt functions for verifying principal components and variance. 
        """
        X = scale(self._X)
        pca = PCA(n_components=13)
        pca.fit_transform(X)
        
        # Cumulative Variance explains
        var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        print "\nCumulative Variance is: ", var1
        
        first_pc = pca.components_[0]
        second_pc = pca.components_[1]
        third_pc = pca.components_[2]
        fourth_pc = pca.components_[3]
        fifth_pc = pca.components_[4]
        
        print "\nFirst principal component: \n", first_pc
        print "\nSecond principal component: \n", second_pc
        print "\nThird principal component: \n", third_pc
        print "\nFourth principal component: \n", fourth_pc
        print "\nFifth principal component: \n", fifth_pc
        
if __name__ == '__main__':
    app = App(sys.argv[1])
    sys.exit(app.computeResults())
