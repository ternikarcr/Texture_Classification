#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:06:36 2020

@author: chirag
"""
####Import required packages####
import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import pylab as pl
import seaborn as sns
from sklearn import decomposition
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from spectres import spectres
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
from sklearn.svm import SVC
svm = SVC()

####Global data FINAL for PAPER####
#Setting up of working directory and loading data
pwd
cd /home/chirag/Documents/HSI/Soil
pwd

df =  pd.read_csv('VNIR1.csv')
df =  pd.read_csv('MIR1.csv')
df =  pd.read_csv('VNIR_MIR.csv')

##Global Datasets
#All Bands
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'Batch_Labid'], axis=1))
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'SSN'], axis=1))
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'Batch_Labid', 'SSN'], axis=1))

##VNIR
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df =  pd.read_csv('VNIR.csv')
name = 'Final_Text_VNIR.txt'
#All Bands
group = 'All Bands'
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#20 Group
group = '20 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1410','W1480','W1810','W1860','W1880','W1900','W1930','W2010','W2100','W2170','W2200','W2270','W2390','W2420','W2440'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##MIR
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df =  pd.read_csv('MIR.csv')
name = 'Final_Text_MIR.txt'
#All Bands
group = 'All Bands'
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##20 groups
group = '20 Group'
fruits = pd.DataFrame(df, columns= ['Texture', 'm705.8','m802.3','m894.8','m919.9','m1110.8','m1213','m1382.7','m1550.5','m1650.8','m1720.2','m1843.6','m1891.8','m1911.1','m2239','m2242.8','m2347','m2476.2','m2684.5','m2738.5','m2877.3','m3062.4','m3077.9','m3249.5','m3257.2','m3587','m3758.6','m3822.3','m3903.3','m3928.3'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##VNIR_MIR
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df =  pd.read_csv('VNIR_MIR.csv')
name = 'Final_Text_VNIR_MIR.txt'
#All Bands
group = 'All Bands'
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##20 groups
group = '20 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1410','W1480','W1810','W1860','W1880','W1900','W1930','W2010','W2100','W2170','W2200','W2270','W2390','W2420','W2440','m705.8','m802.3','m894.8','m919.9','m1110.8','m1213','m1382.7','m1550.5','m1650.8','m1720.2','m1843.6','m1891.8','m1911.1','m2239','m2242.8','m2347','m2476.2','m2684.5','m2738.5','m2877.3','m3062.4','m3077.9','m3249.5','m3257.2','m3587','m3758.6','m3822.3','m3903.3','m3928.3'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

