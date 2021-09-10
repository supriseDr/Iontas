import numpy as np
from fractional_brownian import FBM
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
AUDCAD = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/AUDCAD_H4.csv',  sep='\t')
AUDCHF = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/AUDCHF_H4.csv',  sep='\t')
AUDJPY = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/AUDJPY_H4.csv',  sep='\t')
AUDNZD = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/AUDNZD_H4.csv',  sep='\t')
AUDUSD = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/AUDUSD_H4.csv',  sep='\t')
CADJPY = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/CADJPY_H4.csv',  sep='\t')
CHFJPY = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/CHFJPY_H4.csv',  sep='\t')
EURAUD = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/EURAUD_H4.csv',  sep='\t')
EURCAD = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/EURCAD_H4.csv',  sep='\t')
EURCHF = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/EURCHF_H4.csv',  sep='\t')
EURGBP = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/EURGBP_H4.csv',  sep='\t')
#FOr now data reference to Any instead of Array and Pandas DataType
df = pd.concat([AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP], axis=1)
#returns = utils.logreturn(df['Close'])
#print(returns)
#print(df['Close'])
"""
Download Stocks Closing Price for Companies
"""


from yahoo_fin.stock_info import get_data, tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table
#tgt = get_data("TGT") #Target Corporation
#low = get_data("LOW") #Lowe's companies Corporation
fsr = get_data("FSR.JO") #First Rand
inp = get_data("INP.JO") #Investec Group
mtn = get_data("MTN.JO") #MTN Group
aeg = get_data("AEG.JO") #Aveng Limited
#print(fsr)
"""
nflx = get_data("NFLX")
crwd = get_data("CRWD")
baba = get_data("BABA")
jd = get_data("JD")
goev = get_data("GOEV")
bby = get_data("BBY")
tnxp = get_data("TNXP")
pdd = get_data("PDD")
pbth = get_data("BPTH")
pfe = get_data("PFE")
amc = get_data("AMC")
amd = get_data("AMD")
edu = get_data("EDU")
aapl = get_data("AAPL")
f = get_data("F")
nvda = get_data("NVDA")
nio = get_data("NIO")
tal = get_data("TAL")
t = get_data("T")
msft = get_data("MSFT")
intc = get_data("INTC")
roxif = get_data("ROXIF")
vobif = get_data("VOBIF")
tsp = get_data("TSP")
docs = get_data("DOCS")
stem = get_data("STEM")
ci = get_data("CI")
"""

"""
Data Frame
"""

"""
data = pd.concat([ nflx['close'],crwd['close'],baba['close'] ,jd['close'] ,goev['close'] ,bby['close'] ,tnxp['close'] ,pdd['close'] ,pbth['close'] ,pfe['close'],amc['close'], amd['close'],  edu['close'],  aapl['close'],  f['close'],  nvda['close'],  nio['close'],  tal['close'],  t['close'],  msft['close'],  intc['close'],  roxif['close'],  vobif['close'],  tsp['close'],  docs['close'],  stem['close'],  ci['close']
 ], 

                 axis=1
                 #keys= ['Netflix','CrowdStrike Holdings', 'Alibaba','Palo Alto Networks Inc.', 'Alibaba Group Holding Limited', 'JD.com, Inc.', 'Canoo Inc.', 'Best Buy Co., Inc.', 'Tonix Pharmaceuticals Holding Corp.','Pinduoduo Inc.','Bio-Path Holdings, Inc.','Pfizer Inc.','AMC Entertainment Holdings, Inc.','Advanced Micro Devices, Inc.','New Oriental Education & Technology Group Inc.','Ford Motor Company','NVIDIA Corporation','NIO Inc.','TAL Education Group','AT&T co','Microsoft','Intel','Caspian Sunrise plc','Vobile Group Limited','TuSimple Holdings Inc.','Doximity, Inc.','Stem Inc','Cigna Corporation']
                 )
"""

data = pd.concat([ fsr['close'],inp['close'],mtn['close'],aeg['close']
 ], 
                 axis=1,
                 keys= ['First Rand','Investec Group','MTN Group','Aveng Limited']
                 )


"""
#Save data as csv
data.to_csv(r'~/Desktop/Iontas/Iontas/Data/stockscloseprices.csv')
"""
data.to_csv(r'~/Desktop/Iontas/Iontas/Data/sastockscloses.csv')
#data = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/stockscloseprices.csv')
#data.drop('Unnamed: 0', inplace=True, axis=1)

#data = pd.read_csv(r'~/Desktop/Iontas/Iontas/Data/lowetargetcloses.csv')
#data.drop('Unnamed: 0', inplace=True, axis=1)

#data

#returns = utils.logreturn(data)
#print(utils.describedata(returns))
#print(utils.returnsvariance(returns))
#print(returns)
"""
Correlations
"""
#correlation = utils.returnscorrelation(returns)
#print(covariance)

#covariance = utils.returnscovariance(returns)

#print(covariance)
#utils.heatmap(correlation, True)

#print spectrogram
#eigenvalues = utils.eigen(correlation)
#eigenvalues = utils.eigen(covariance)
#Print Eigen Values then Compare ALgorithms
#utils.eigenvalues(correlation)
#near = utils.nearcorrelation(covariance)

#neareigenvalues = utils.eigen(correlation)
#neareigenvalues = utils.eigen(near)

#print(covariance)
#print(eigenvalues)
#print(near)
#print(neareigenvalues)

#utils.heatmap(near, False)

#print(neareigenvalues)
"""
Show Plots
"""
plt.show()

#utils.spectralDecompose(correlation)

#utils.nearcorrelation(A)
#f = FBM(n=1024, hurst=0.75, length=1, method='daviesharte')
#print(f)

#import metrics as mt
#points_a = (3, 0)
#points_b = (6, 0)

#print(mt.hausdorff_distance(points_a, points_b))