import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from plotly.graph_objs import *


class App():
    def __init__(self, filename):
    	print 'filename is "%s"' % filename
        
        # Loads the data
        self._leaf = pd.read_csv('/Users/shankartiwari/Desktop/leaf.csv', dtype=np.float64)
       
        # Spliting the leaf data into training data and label.
        self._X = self._leaf.ix[:,3:15].values
        self._y = self._leaf.ix[:,0].values

        # Variable used for standardization
        self._standardizeVariable = None

        # Mean vector variable
        self._meanVector = None

        # Covariance Matrix variable
        self._covarianceMatrix = None

        # Eigen vector and eigen values
        self._eigenVectors = None
        self._eigenValues = None
        self._eigPairs = []

        # Variance explanation variable
        self._varExp = []


    def computeVariance(self):
        """
	    Standardize the data by transforming it onto a unit scale (mean=0 and variance=1).
	    Performing pca on unnormalized data will lead to dependence on variable with high variance. 
	    Also pca can be performed only on numerical data.
	    """
	    self._standardizeVariable = StandardScaler().fit_transform(self._X)
	    
	    # Covariance matrix -> Covariance matrix represents the covariance between different features in dataset.
        self._meanVector = np.mean(self._standardizeVariable, axis=0)
        self._covarianceMatrix = (self._standardizeVariable - self._meanVector).T.dot((self._standardizeVariable - self._meanVector)) / (self._standardizeVariable.shape[0]-1)

        """
	    Implementing a eigendecomposition on covariance matrix gives eigen values and eigen vectors.
	    Sum of the eigen values gives the variance in the dataset.
	    """
        self._eigenValues, self._eigenVectors = np.linalg.eig(self._covarianceMatrix)

        print('Computed Eigenvectors are: \n%s' %self._eigenVectors)
        print('\nComputed Eigenvalues are: \n%s' %self._eigenValues)

	    # Make a list of (eigenvalue, eigenvector) tuples
	    
        for i in range(len(self._eigenValues)):
	       self._eigPairs.append((np.abs(self._eigenValues[i]), self._eigenVectors[:,i]))
	   
	    # Sort the (eigenvalue, eigenvector) tuples from high to low
        self._eigPairs.sort()
        self._eigPairs.reverse()

        tot = sum(self._eigenValues)

        for i in sorted(self._eigenValues, reverse=True):
            self._varExp.append((i / tot)*100)

        cumVarExp = np.cumsum(self._varExp)

        print('\nFinal Cummlative Variance is: \n%s' %cumVarExp)
        

    def verifyCalculation(self):

        # Scaling the values
        X = scale(X)

        pca = PCA(n_components=12)

        pca.fit(X)

        # The amount of variance that each PC explains
        var = pca.explained_variance_ratio_

        # Cumulative Variance explains
        var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
	    #plt.plot(var1)

if __name__ == '__main__':
    app = App(sys.argv[1])
    for arg in sys.argv:
    	print arg
    sys.exit(app.computeVariance())