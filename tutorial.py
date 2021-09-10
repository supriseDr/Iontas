"""
Machine learning tools
"""
from numpy.core.fromnumeric import mean
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import DataStream

"""
Import Utils functions
"""
from utils import Utils
import utils

"""
Use Pandas To read csv data
"""
import pandas as pd

"""
Import For Plot Visualization
"""
import matplotlib.pyplot as plt

"""
Stats models for graphic visuals
"""
from statsmodels.graphics.tsaplots import plot_acf

"""
Load CSV local file of stocks closing prices data , then remove the unnamed column
"""
data = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/sastockscloses.csv')
data.drop('Unnamed: 0', inplace=True, axis=1)

"""
Initialize module
utils.Utils
"""
Utils = Utils(data)
"""
Compute logarithmic returns
"""

returns = Utils.logreturn() 
import numpy as np
mean = np.mean(returns)
variance = np.var(returns)
#plt.plot(variance, mean, 'ro')
#plt.ylabel('Expected Return')
#plt.xlabel('Variance')

normreturns = Utils.normalizedreturns()

correlations = Utils.returnscorrelation()

variances = Utils.returnsvariance()

covariances = Utils.returnscovariance()

descriptive = Utils.describedata(returns)

nearcorrelation = utils.nearcorrelation(covariances)

eigen = Utils.eigen(correlations)

#Utils.histogram_density(X)

"""
Spectrogram Plot
"""
"""
from scipy import signal
freqs, times, spectrogram = signal.spectrogram(X)

plt.figure(figsize=(5, 4))
plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.title('MTN Group Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()
"""
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_trisurf(X, Y, Z, cmap=plt.cm.jet, linewidth=0.01)

#print(nearcorrelation)
"""Import nx graph module to work with Graphs"""
import networkx as nx
#G = nx.Graph(correlations)
#nx.draw(G, with_labels=True) # The Graph Structure
#nx.draw_shell(G, with_labels=True) # The Graph Structure with shell layout reveals industries
#nx.draw_spring(G, with_labels=True) # The Graph Structure with spring layout reveals relations
#nx.draw_spectral(G, with_labels=True) # The 2D spectral Structure


"""
stream = returns['Netflix']
stream = DataStream(returns['Netflix'])
ht = HoeffdingTreeClassifier()
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=1000,
                                max_samples=10000,
                                output_file='results.csv')

evaluator.evaluate(stream=stream, model=ht)
"""

"""
Compute pairwise correlation using the Pearson method
"""
#correlation = Utils.returnscorrelation()
#print(correlation)

#utils.heatmap(correlation, False)

#eigenvalues = Utils.eigen()
#print(eigenvalues)

"""
Compute pairwise covariance matrix
"""
#covariance =Utils.returnscovariance()
#print(covariance)

#coveigenvalues = Utils.eigen(covariance)
#print(coveigenvalues)

#nearcorrmatrix = Utils.nearcorrelation(covariance)
#print(nearcorrmatrix)

#nearcorrmatrixeigenvalues = Utils.eigen(nearcorrmatrix)
#print(nearcorrmatrixeigenvalues)

"""
Plot autocorrelation with 50 lags
"""
#plot_acf(x=returns['Doximity, Inc.'], lags=1)

"""
Show Plots
"""
plt.show()
