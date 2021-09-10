"""
The utils file consist of common functions perfomed by the software
-Log returns
-Correlation Matrix
"""
#import warnings
import numpy as np

from numpy.lib.function_base import copy
from numpy.linalg.linalg import norm
import seaborn as sns; sns.set_theme() #Use seaborn for heat map
from numpy import inf, linalg as LA #for matrix eigen values calculation
from scipy import signal
import matplotlib.pyplot as plt

"""
The common functions need to be contained under the utils class
"""

class Utils():
    
    def __init__(self, stocks_data):
        #self._methods = {"pearson"} for correlation methods
        self._data = stocks_data 

    """
    Log returns Function Using numpy
    The log of price security current less previous
    -: ln(Ps(t + delta t))- ln(Ps(t))
    -: args X: Stocks Matrix or Pandas DataFrame
    """
    def logreturn(self):
        return np.log(self._data) - np.log(self._data.shift(1))

    """
    Returns Histogram
    """
    def histogram(X):
        return plt.hist(X)

    """
    Normalized Log returns Function Using numpy Nested for Normalization
    The log of price security current less previous
    -: ln(Ps(t + delta t))- ln(Ps(t))/std
    -: args X: Stocks Matrix or Pandas DataFrame
    """
    def normalizedreturns(self):
        logreturn = self.logreturn()
        return logreturn/logreturn.std()
    """
    The correlation Matrix takes as Input the Array of Logreturn Assets
    -: args X: Stocks Log returns
    """
    def returnscorrelation(self):
        return self.logreturn().corr(method='pearson')

    """
    The variance vector of the stock returns
    -: args X: Stocks Log returns
    """
    def returnsvariance(self):
        return self.logreturn().var()

    """
    The covariance matrix of the stock returns
    -: args X: Stocks Log returns
    """
    def returnscovariance(self):
        return self.logreturn().cov()

    """
    Pandas Describe data series
    """
    def describedata(self, X):
        return X.describe()


    """
    Construct a Heat Map Representation of The correlation Matrix
    -: args X: The correlation matrix
            Bool: Annotation
    """
    def heatmap(self , X, boolean):
        ax = sns.heatmap(X, annot=boolean)
        return ax
    
    """
    Density plots and histograms
    """
    def histogram_density(self, X):
        return sns.distplot(a=X)
    
    """
    Compute Eigen Values of a correlation matrix
    -: args X: The correlation matrix
    """
    def eigen(self, X):
        w, v = LA.eig(X)
        return w , v
        #return v
        #freqs, times, spectrogram = signal.spectrogram(v[0])
        #plt.figure(figsize=(5, 4))
        #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
        #plt.title('Spectrogram')
        #plt.ylabel('Frequency band')
        #plt.xlabel('Time window')
        #plt.tight_layout()

    """
    Compute Eigen Values of a correlation matrix
    -: args X: The correlation symmetric matrix
    """
    def eigenvalues(self, X):
        print(LA.eigvalsh(X))
    

    """
    Compute Eigen Values of a correlation matrix
    -: args X: The correlation symmetric matrix
    """   
def spectralDecompose(X):
    d, v = LA.eigh(X)
    X = (v*np.maximum(d, 0)).dot(v.T)
    X = (X + X.T)/2
    #print(X)
    return(X)

    """Exceeded Max iterations class"""
class ExceededMaxIterationsError(Exception):
    def __init__(self, msg, matrix=[], iteration=[], ds=[]):
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self):
            return repr(self.msg)

    """
    Compute near correlation matrix
    -: args X: The correlation symmetric matrix using Alternate Projective Method
    """   
def nearcorrelation(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
                weights=None, verbose=False,
                except_on_too_many_iterations=True):
        ds = np.zeros(np.shape(A))
        eps = np.spacing(1) #Replacer for negative values
        
        if not np.all((np.transpose(A) == A )):
            raise ValueError('Input Matrix is not symmetric')
        if not tol:
            tol = eps*np.shape(A)[0]*np.array([1, 1]) #convergence tolerance
        if weights is None:
            weights = np.ones(np.shape(A)[0])
        X = copy(A)
        Y = copy(A)
        rel_diffY = inf
        rel_diffX = inf
        rel_diffXY = inf
        
        Whalf = np.sqrt(np.outer(weights, weights)) #Computes the outer products of two vectors
        iteration = 0
        while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
            iteration += 1
            if iteration > max_iterations:
                if except_on_too_many_iterations:
                    if max_iterations == 1:
                        message = "No solution found in "\
                                + str(max_iterations) + " iteration"
                    else:
                        message = "No solution found in "\
                                + str(max_iterations) + " iterations"
                    raise ExceededMaxIterationsError(message, X, iteration, ds)
                else:
                    # exceptOnTooManyIterations is false so just silently
                    # return the result even though it has not converged
                    return X

            Xold = copy(X)
            R = X - ds
            R_wtd = Whalf*R
            if flag == 0:
                X = spectralDecompose(R_wtd)
            elif flag == 1:
                raise NotImplementedError("Setting 'flag' to 1 is currently\
                                    not implemented.")
            X = X / Whalf
            ds = X - R
            Yold = copy(Y)
            Y = copy(X)
            np.fill_diagonal(Y, 1)
            normY = norm(Y, 'fro')
            rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
            rel_diffY = norm(Y - Yold, 'fro') / normY
            rel_diffXY = norm(Y - X, 'fro') / normY

            X = copy(Y)

        #print(X)
        return X
        #print(eps)  

"""
Autocorrelation computations
np.correlate(x, x, mode=full)
"""