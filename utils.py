"""
The utils file consist of common functions perfomed by the software
-Log returns
-Correlation Matrix
"""
import numpy as np
import seaborn as sns; sns.set_theme() #Use seaborn for heat map
from numpy import linalg as LA #for matrix eigen values calculation
from scipy import signal
import matplotlib.pyplot as plt
"""
Log returns Function Using numpy
The log of price security current less previous
-: ln(Ps(t + delta t))- ln(Ps(t))
-: args X: Stocks Matrix or Pandas DataFrame
"""
def logreturn(X):
    return np.log(X) - np.log(X.shift(1))

"""
The correlation Matrix takes as Input the Array of Logreturn Assets
-: args X: Stocks Log returns
"""
def returnscorrelation(X):
    return X.corr(method='pearson')

"""
Construct a Heat Map Representation of The correlation Matrix
-: args X: The correlation matrix
        Bool: Annotation
"""
def heatmap(X, boolean):
    ax = sns.heatmap(X, annot=boolean)
    return ax
"""
Compute Eigen Values of a correlation matrix
-: args X: The correlation matrix
"""
def eigen(X):
    w, v = LA.eig(X)
    print(w , v)
    freqs, times, spectrogram = signal.spectrogram(v[0])
    plt.figure(figsize=(5, 4))
    plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    plt.title('Spectrogram')
    plt.ylabel('Frequency band')
    plt.xlabel('Time window')
    plt.tight_layout()

"""
Compute Eigen Values of a correlation matrix
-: args X: The correlation symmetric matrix
"""
def eigenvalues(X):
    print(LA.eigvalsh(X))

    