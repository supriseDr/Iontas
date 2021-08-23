"""
Import Utils functions
"""
import utils
"""
Use Pandas DataReader to Read CSV data
"""
import pandas as pd
"""
Import Plot Visualization
"""
import matplotlib.pyplot as plt
#use \ when working on windows system
AUDCAD = pd.read_csv(r'~/Desktop/Iontas/Data/AUDCAD_H4.csv',  sep='\t')
AUDCHF = pd.read_csv(r'~/Desktop/Iontas/Data/AUDCHF_H4.csv',  sep='\t')
AUDJPY = pd.read_csv(r'~/Desktop/Iontas/Data/AUDJPY_H4.csv',  sep='\t')
AUDNZD = pd.read_csv(r'~/Desktop/Iontas/Data/AUDNZD_H4.csv',  sep='\t')
AUDUSD = pd.read_csv(r'~/Desktop/Iontas/Data/AUDUSD_H4.csv',  sep='\t')
CADJPY = pd.read_csv(r'~/Desktop/Iontas/Data/CADJPY_H4.csv',  sep='\t')
CHFJPY = pd.read_csv(r'~/Desktop/Iontas/Data/CHFJPY_H4.csv',  sep='\t')
EURAUD = pd.read_csv(r'~/Desktop/Iontas/Data/EURAUD_H4.csv',  sep='\t')
EURCAD = pd.read_csv(r'~/Desktop/Iontas/Data/EURCAD_H4.csv',  sep='\t')
EURCHF = pd.read_csv(r'~/Desktop/Iontas/Data/EURCHF_H4.csv',  sep='\t')
EURGBP = pd.read_csv(r'~/Desktop/Iontas/Data/EURGBP_H4.csv',  sep='\t')
#FOr now data reference to Any instead of Array and Pandas DataType
df = pd.concat([AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP], axis=1)
returns = utils.logreturn(df['Close'])
#print(returns)
#print(df['Close'])
"""
Correlations
"""
correlation = utils.returnscorrelation(returns)
#print(correlation)

#utils.heatmap(correlation, True)

#print spectrogram
#utils.eigen(correlation)
#Print Eigen Values then Compare ALgorithms
utils.eigenvalues(correlation)
"""
Show Plots
"""
#plt.show()
