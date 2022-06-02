#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:06:36 2020

@author: chirag
"""
import sys
import os
import glob

pwd
cd /home/chirag/Documents/HSI/Soil
cd /home/chirag/Documents/HSI/Soil/Global
cd /home/chirag/Documents/HSI/Soil/Global/Refined
cd /home/chirag/Documents/HSI/Soil/Beginning
cd /home/chirag/Documents/HSI/Soil/Beginning/Unmix_Trial
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
cd /home/chirag/Documents/HSI/Soil/Bands/Regression

pwd
ls

#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler = MinMaxScaler()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import precision_recall_fscore_support as prfs
from spectres import spectres
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import pylab as pl
import seaborn as sns
from sklearn import decomposition
import collections

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#from sklearn.tree import DecisionTreeClassifier
#dtc = DecisionTreeClassifier()
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
#from sklearn.naive_bayes import GaussianNB
#gnb = GaussianNB()
from sklearn.svm import SVC
svm = SVC()

#df = pd.read_excel('IP.xlsx')
#df = pd.read_excel('IP7.xlsx')
#df = pd.read_excel('IP6.xlsx')
#df = pd.read_excel('IP5.xlsx')
#df = pd.read_excel('IP4.xlsx')
#df = pd.read_excel('IP3.xlsx')
#df = pd.read_excel('IP2.xlsx')
#df = pd.read_excel('IP1.xlsx')
#df = pd.read_excel('Global_spectra.xlsx')

#df1 =  pd.read_csv('Country.csv')
#df2 =  pd.read_csv('Physical_properties.csv')
#df1 =  pd.read_csv('ICRAF_ISRIC_reference_data.csv')
#df =  pd.read_csv('Temp.csv')
#df1 =  pd.read_csv('ICRAF_sample_codes.csv')
#df =  pd.read_csv('Temp1.csv')
#df =  pd.read_csv('Temp_with_codes.csv')
#df1 =  pd.read_csv('ASD_Spectra.csv')
#df2 =  pd.read_csv('ICRAF_ISRIC_MIR_spectra1.csv')

df =  pd.read_csv('ASD_Spectra_Master.csv')
df =  pd.read_csv('ICRAF_ISRIC_MIR_spectra1_Master.csv')
df =  pd.read_csv('ICRAF_ISRIC_VNIR_MIR_spectra_Master.csv')

df =  pd.read_csv('VNIR1.csv')
df =  pd.read_csv('MIR1.csv')
df =  pd.read_csv('VNIR_MIR.csv')

df =  pd.read_csv('VNIR1.csv')
df =  pd.read_csv('MIR1.csv')
df =  pd.read_csv('VNIR_MIR.csv')
df =  pd.read_csv('VNIR_MIR_common_rm_MIR.csv')
df =  pd.read_csv('VNIR_MIR_common_rm_VNIR.csv')


####DATA WRANGLING and Duplicate checking####
df['SAMPLENO'] = df['SAMPLENO'].astype(int)
df1['Sampleno'] = df1['Sampleno'].astype(int)

result = pd.merge(df, df1, left_on = 'Batch_labid', right_on = 'Batch_Labid', how = 'inner')
result1 = pd.merge(result, df2, left_on = 'Batch_labid', right_on = 'SSN', how = 'inner')
result.shape
result.drop_duplicates(subset="SAMPLENO", keep=False, inplace=True)
result.shape
result1.to_csv('ICRAF_ISRIC_VNIR_MIR_spectra_Master.csv')
result1.shape

df.shape
df1.shape
a = result['Sampleno']
b = result['Sampleno']
sum(a == b)
a.drop_duplicates(keep=False, inplace=True)
duplicate = b[b.duplicated(keep=False)] 

print (df)
df.shape
#We have 274 points and 260 spectral bands
df.columns
df1 = df['Batch_labid'].copy()
df2 = df1['Sampleno'].copy()
#df1 = df1.drop(['Sample_ID', 'Texture', 'Collector', 'Texture_num'], axis=1)
#df1 = df[df.columns.difference(['Sample_ID', 'Texture', 'Collector', 'Texture_num'])]
df1.shape
duplicate = df1[df1.duplicated(keep=False)] 
duplicate.shape
del df2,df


df2 = df1.copy()
df2 = df2.drop(['SSN'], axis=1)
df2.drop_duplicates(keep=False, inplace=True)
df2 = df2.drop(['Unnamed: 0'], axis=1)
df2 = df2.drop(['m7498', 'm599.8'], axis=1)
df2.to_csv('ICRAF_ISRIC_MIR_spectra1.csv')



#All Bands
#fruits = pd.DataFrame(df.drop(['Sample_ID', 'Collector', 'Texture_num'], axis=1))
#fruits = pd.DataFrame(df.drop(['Sample_ID', 'Collector', 'Texture_num', 'NDVI'], axis=1))
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

#df = pd.read_excel('IP.xlsx')
#df = pd.read_excel('IP7.xlsx')
#df = pd.read_excel('IP6.xlsx')
#df = pd.read_excel('IP5.xlsx')
#df = pd.read_excel('IP4.xlsx')
#df = pd.read_excel('IP3.xlsx')
#df = pd.read_excel('IP2.xlsx')
#df = pd.read_excel('IP1.xlsx')
#No grouping
#fruits = pd.DataFrame(df, columns= ['Texture','band_67'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_67'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_65'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_65'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_42','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_42','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_43','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_58','band_94','band_111'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

#df = pd.read_excel('IP.xlsx')
#df = pd.read_excel('IP7.xlsx')
#df = pd.read_excel('IP6.xlsx')
#df = pd.read_excel('IP5.xlsx')
#df = pd.read_excel('IP4.xlsx')
#df = pd.read_excel('IP3.xlsx')
#df = pd.read_excel('IP2.xlsx')
#df = pd.read_excel('IP1.xlsx')
#5 groups 
#fruits = pd.DataFrame(df, columns= ['Texture','band_67','band_69','band_192','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_67','band_69','band_190','band_260','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_67','band_69','band_190','band_260','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_67','band_69','band_190','band_260','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_67','band_69','band_190','band_244','band_274','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_65','band_69','band_190','band_252','band_274','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_62','band_69','band_72','band_192','band_242','band_271','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_58','band_73','band_163','band_190','band_243','band_248','band_257','band_259','band_267','band_269','band_272','band_275','band_327','band_335','band_388'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

#df = pd.read_excel('IP.xlsx')
#df = pd.read_excel('IP7.xlsx')
#df = pd.read_excel('IP6.xlsx')
#df = pd.read_excel('IP5.xlsx')
#df = pd.read_excel('IP4.xlsx')
#df = pd.read_excel('IP3.xlsx')
#df = pd.read_excel('IP2.xlsx')
#df = pd.read_excel('IP1.xlsx')
#10 groups
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_67','band_69','band_166','band_192','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_67','band_69','band_163','band_190','band_260','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_42','band_65','band_69','band_163','band_192','band_244','band_260','band_328','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_65','band_69','band_131','band_163','band_192','band_244','band_260','band_361','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_67','band_69','band_131','band_163','band_190','band_244','band_274','band_361','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_65', 'band_69','band_131','band_163','band_190','band_238','band_245','band_250','band_252','band_274','band_361','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_61', 'band_69', 'band_72','band_134','band_162', 'band_175','band_192','band_240','band_271','band_328','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_58', 'band_59', 'band_73','band_140','band_142','band_163', 'band_182', 'band_224', 'band_227', 'band_231','band_241', 'band_248', 'band_257', 'band_259','band_267', 'band_269', 'band_272','band_275','band_361', 'band_388'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

#df = pd.read_excel('IP.xlsx')
#df = pd.read_excel('IP7.xlsx')
#df = pd.read_excel('IP6.xlsx')
#df = pd.read_excel('IP5.xlsx')
#df = pd.read_excel('IP4.xlsx')
#df = pd.read_excel('IP3.xlsx')
#df = pd.read_excel('IP2.xlsx')
#df = pd.read_excel('IP1.xlsx')
#15 groups
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_48','band_67','band_69','band_166','band_173','band_192','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_53','band_67','band_69','band_163','band_176','band_190','band_259','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_53','band_65','band_69','band_163','band_173','band_192','band_238','band_259','band_328','band_366','band_373'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_53','band_65','band_69','band_94','band_131','band_163','band_172','band_192','band_238','band_259','band_328','band_366','band_373'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_53','band_67','band_69','band_92','band_115','band_163','band_174','band_190','band_238','band_259','band_274','band_328','band_330','band_366','band_373'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_53','band_65','band_69','band_94','band_96','band_115','band_163','band_181','band_190','band_238','band_251','band_259','band_274','band_328','band_366','band_373'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_53','band_61','band_69','band_72','band_92','band_96','band_134','band_168','band_180','band_192','band_235','band_246','band_271','band_328','band_366','band_373'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_48','band_51','band_58','band_73','band_94','band_96','band_99','band_101','band_105','band_108','band_110','band_163','band_176','band_182','band_233','band_239','band_247','band_256','band_263','band_269','band_272','band_275','band_329','band_333','band_248','band_361','band_388'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

#df = pd.read_excel('IP.xlsx')
#df = pd.read_excel('IP7.xlsx')
#df = pd.read_excel('IP6.xlsx')
#df = pd.read_excel('IP5.xlsx')
#df = pd.read_excel('IP4.xlsx')
#df = pd.read_excel('IP3.xlsx')
#df = pd.read_excel('IP2.xlsx')
#df = pd.read_excel('IP1.xlsx')
#20 groups
#fruits = pd.DataFrame(df, columns= ['Texture','band_45','band_48','band_67','band_69','band_166','band_171','band_185','band_192','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_45','band_53','band_67','band_69','band_163','band_171','band_185','band_190','band_249','band_260','band_266','band_366'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_34','band_45','band_53','band_65','band_69','band_163','band_171','band_185','band_192','band_234','band_247','band_260','band_266','band_328','band_360','band_366','band_394'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_34','band_45','band_53','band_65','band_69','band_85','band_115','band_131','band_163','band_171','band_185','band_192','band_234','band_247','band_260','band_266','band_328','band_360','band_366','band_394'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_34','band_45','band_53','band_67','band_69','band_90','band_103','band_114','band_131','band_163','band_170','band_185','band_190','band_234','band_247','band_260','band_274','band_328','band_330','band_259','band_366','band_383'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_34','band_45','band_56','band_65','band_69','band_94','band_96','band_103','band_114','band_131','band_163','band_174','band_185','band_190','band_234','band_248','band_262','band_264','band_274','band_328','band_359','band_366','band_394'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_45','band_56','band_61','band_69','band_72','band_89','band_102','band_134','band_168','band_175','band_185','band_192','band_232','band_246','band_264','band_271','band_328','band_359','band_366','band_394'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_45','band_55','band_58','band_73','band_94','band_103','band_109','band_140','band_142','band_163','band_176','band_182','band_186','band_226','band_230','band_247','band_259','band_264','band_267','band_272','band_276','band_329','band_333','band_342','band_354','band_372','band_375','band_377','band_388'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


#Foo_All Dataset contains 133 points with NDVI < 0.3 and band_350 > band_365
#df = pd.read_excel('Foo_All.xlsx')
#All Bands
#fruits = pd.DataFrame(df.drop(['Sample_ID', 'Strip', 'Collector', 'Texture_num'], axis=1))
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#No grouping
#fruits = pd.DataFrame(df, columns= ['Texture','band_373'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#5 groups 
#fruits = pd.DataFrame(df, columns= ['Texture','band_58','band_69','band_190','band_245','band_252','band_274','band_373'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#10 groups 
#fruits = pd.DataFrame(df, columns= ['Texture','band_44','band_58','band_69','band_131','band_136','band_177','band_190','band_235','band_245','band_252','band_274','band_361','band_373'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#15 groups
#fruits = pd.DataFrame(df, columns= ['Texture','band_38','band_53','band_58','band_69','band_94','band_96','band_115','band_168','band_181','band_190','band_235','band_242','band_253','band_274','band_328','band_366','band_373'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#20 groups
#fruits = pd.DataFrame(df, columns= ['Texture','band_45','band_56','band_58','band_69','band_88','band_103','band_131','band_136','band_168','band_177','band_185','band_190','band_234','band_264','band_274','band_328','band_359','band_373','band_394'])
#fruits = pd.DataFrame(df, columns= ['Texture','band_45','band_56','band_58','band_69','band_88','band_103','band_131','band_136','band_168','band_177','band_185','band_190','band_234','band_235','band_264','band_274','band_328','band_359','band_373','band_394'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


#BM Dataset contains 275 points from lab spectra
df3 = pd.read_excel('BM.xlsx')
df = pd.read_excel('BM_spectra.xlsx')

#df1 = df.copy()
#df1 = df1.drop(['Sample_ID', 'Texture'], axis=1)
#df1 = df[df.columns.difference(['Sample_ID', 'Texture'])]
#df1.shape
#duplicate = df1[df1.duplicated(keep=False)] 
#duplicate.shape
#del df1,duplicate


#All Bands
fruits = pd.DataFrame(df.drop(['Sample_ID'], axis=1))
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#No grouping
#fruits = pd.DataFrame(df, columns= ['Texture','band_2500'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#5 groups 
#fruits = pd.DataFrame(df, columns= ['Texture','band_715','band_800','band_1367','band_1416','band_2500'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#10 groups 
#fruits = pd.DataFrame(df, columns= ['Texture','band_715','band_800','band_1000','band_1367','band_1416','band_1831','band_1849','band_1913','band_2000','band_2003','band_2212','band_2500'])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#15 groups
#fruits = pd.DataFrame(df, columns= ["Texture", "band_649",  "band_715",  "band_800",  "band_1000", "band_1152", "band_1416", "band_1367",  "band_1500", "band_1509", "band_1849", "band_1831", "band_1913", "band_2000", "band_2003",  "band_2199", "band_2212", "band_2399", "band_2341", "band_2500"])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#20 groups
#fruits = pd.DataFrame(df, columns= ["Texture", "band_649",  "band_715",  "band_800",  "band_900",  "band_1000", "band_1102", "band_1200",  "band_1399", "band_1372", "band_1416", "band_1499", "band_1500", "band_1509", "band_1699",  "band_1799", "band_1770", "band_1899", "band_1913", "band_2000", "band_2003", "band_2199",  "band_2212", "band_2399", "band_2341", "band_2500"])
#runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')



##Master Files Creation
df1 =  pd.read_excel('Aviris_spectra.xlsx')
df2 =  pd.read_excel('BM_spectra.xlsx')
df3 =  pd.read_excel('Aviris_ASD_spectra_Master.xlsx')

result = pd.merge(df1, df2, left_on = 'Sample_ID', right_on = 'Sample_ID', how = 'inner')
result.shape
result.to_excel('Aviris_ASD_spectra_Master.xlsx')


#Removing duplicates if any
fruits = pd.DataFrame(df1.drop(['Sample_ID'],axis=1))
fruits = df2.drop(df2.loc[:,'Sample_ID' : 'CBD Fe '].columns,axis=1)
fruits.shape
a = fruits.drop_duplicates()
a.shape
del a,fruits

##Global Datasets
#All Bands
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'Batch_Labid'], axis=1))
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'SSN'], axis=1))
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'Batch_Labid', 'SSN'], axis=1))
fruits = fruits.drop(fruits.loc[:, 'm7496':'m601.7'].columns, axis = 1) 
fruits = fruits.drop(fruits.loc[:, 'm7496':'m4001.6'].columns, axis = 1) 
fruits = fruits.drop(fruits.loc[:, 'W350':'W2500'].columns, axis = 1) 
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

###VNIR
##No Grouping
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","W1420", "W1890", "W1990", "W2340", "W2460"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set Final
fruits = pd.DataFrame(df, columns= ['Texture','W520','W1420', 'W1890', 'W1990', 'W2360', 'W2450', 'W2480'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set & sub set Final
fruits = pd.DataFrame(df, columns= ['Texture','1420', '1990'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##5 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","W1360", "W1410", "W1500", "W1890", "W1990", "W2460"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set Final
fruits = pd.DataFrame(df, columns= ['Texture','W1330', 'W1350', 'W1410', 'W1450', 'W1470', 'W1510', 'W1810', 'W1840', 'W1920', 'W1990', 'W2330', 'W2450'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set and Subset final
fruits = pd.DataFrame(df, columns= ['Texture','1410', '1990','2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##10 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","W1360", "W1410", "W1470", "W1870", "W1890", "W1900", "W1910", "W1920", "W1990", "W2000", "W2020", "W2090", "W2160", "W2210", "W2290", "W2300", "W2360", "W2380", "W2460"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set Final
fruits = pd.DataFrame(df, columns= ['Texture','W1330', 'W1360', 'W1410', 'W1440', 'W1460', 'W1490', 'W1860', 'W1880', 'W1900', 'W1930', 'W1960', 'W2020', 'W2080', 'W2150', 'W2170', 'W2210', 'W2290', 'W2300', 'W2360', 'W2390', 'W2420', 'W2450'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1360', 'W1410', 'W1860', 'W1900', 'W1910', 'W2000', 'W2010', 'W2090', 'W2160', 'W2210', 'W2290', 'W2300', 'W2380', 'W2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set Final
fruits = pd.DataFrame(df, columns= ['Texture','1360', '1410', '1860', '1900', '2010', '2090', '2160', '2210', '2290', '2380', '2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##15 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","W1360", "W1410", "W1470", "W1870", "W1890", "W1900", "W1910", "W1920", "W1990", "W2000", "W2020", "W2090", "W2150", "W2170", "W2190", "W2200", "W2210", "W2270", "W2290", "W2300", "W2360", "W2380", "W2390", "W2400", "W2410", "W2460"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set Final
fruits = pd.DataFrame(df, columns= ['Texture','W1350', 'W1410', 'W1430', 'W1450', 'W1470', 'W1490', 'W1860', 'W1880', 'W1900', 'W1930', 'W2000', 'W2090', 'W2150', 'W2200', 'W2240', 'W2270', 'W2300', 'W2360', 'W2380', 'W2420', 'W2450'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1360', 'W1410', 'W1900', 'W1910', 'W1920', 'W1990', 'W2000', 'W2010', 'W2090', 'W2150', 'W2170', 'W2190', 'W2200', 'W2210', 'W2270', 'W2300', 'W2340', 'W2390', 'W2410', 'W2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set Final
fruits = pd.DataFrame(df, columns= ['Texture','1360', '1410', '1900', '1920', '1990', '2010', '2090', '2150', '2170', '2200', '2270', '2300', '2340', '2390', '2410', '2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##20 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","W1400", "W1410", "W1480", "W1800", "W1870", "W1880", "W1890", "W1910", "W1900", "W1920", "W1990", "W2000", "W2020", "W2090", "W2150", "W2170", "W2190", "W2200", "W2210", "W2270", "W2290" ,"W2300", "W2360", "W2380", "W2390", "W2400", "W2410", "W2460"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set Final
fruits = pd.DataFrame(df, columns= ['Texture', 'W1380', 'W1400', 'W1420', 'W1440', 'W1480', 'W1800', 'W1860', 'W1880', 'W1900', 'W1920', 'W1940', 'W1980', 'W2000', 'W2090', 'W2150', 'W2170', 'W2200', 'W2260', 'W2280', 'W2300', 'W2340', 'W2360', 'W2380', 'W2400', 'W2420', 'W2450', 'W2480'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1400', 'W1410', 'W1480', 'W1800', 'W1870', 'W1880', 'W1890', 'W1900', 'W1910', 'W2000', 'W2010', 'W2090', 'W2150', 'W2170', 'W2190', 'W2200', 'W2210', 'W2270', 'W2290', 'W2300', 'W2390', 'W2410', 'W2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set Final
fruits = pd.DataFrame(df, columns= ['Texture','1410', '1480', '1800', '1880', '1900', '2010', '2090', '2150', '2170', '2200', '2270', '2290', '2390', '2410', '2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


###MIR (1.3 - 16 micrometer)
##No Grouping
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","m3698.8", "m3696.9", "m1810.9", "m1768.4", "m1112.7", "m1110.8",  "m817.7", "m804.2"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set
fruits = pd.DataFrame(df, columns= ['Texture','m806.1', 'm831.2', 'm1110.8', 'm1766.5', 'm1801.2', 'm1822.4', 'm1828.2', 'm1834', 'm1835.9', 'm1868.7', 'm3116.4', 'm3153.1', 'm3698.8'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m1110.8', 'm3696.9'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##5 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","m740.5","m775.3", "m804.2", "m925.7", "m1110.8", "m1835.9", "m1843.6","m1859.1", "m1864.8", "m1999.8", "m2019.1", "m2306.5", "m3625.6", "m3999.7", "m4107.7", "m4186.7", "m5048.8", "m5293.7"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set
fruits = pd.DataFrame(df, columns= ['Texture','m707.8', 'm740.5', 'm784.9', 'm788.8', 'm794.5', 'm802.3', 'm808', 'm831.2', 'm856.2', 'm927.6', 'm966.2', 'm1112.7', 'm1130.1', 'm1253.5', 'm1365.4', 'm1639.2', 'm1770.4', 'm1808.9', 'm1812.8', 'm1826.3', 'm1832.1', 'm1841.7', 'm1853.3', 'm1859.1', 'm1866.8', 'm1872.6', 'm1897.6', 'm1918.8', 'm1928.5', 'm1943.9', 'm1955.5', 'm1959.3', 'm1967.1', 'm1970.9', 'm1990.2', 'm1996', 'm1999.8', 'm2017.2', 'm2026.8', 'm2107.8', 'm2154.1', 'm2169.5', 'm2177.3', 'm2306.5', 'm2840.7', 'm3625.6', 'm3654.5', 'm3700.8', 'm3999.7', 'm4073', 'm4111.5', 'm4132.8', 'm4167.5', 'm4190.6', 'm4242.7', 'm5041.1', 'm5060.4', 'm5066.1', 'm5262.8', 'm5272.5', 'm5291.8'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m856.2', 'm925.7','m1110.8', 'm1999.8', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


##10 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture",])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set
fruits = pd.DataFrame(df, columns= ['Texture','m622.9', 'm632.5', 'm698.1', 'm705.8', 'm732.8', 'm738.6', 'm786.8', 'm794.5', 'm802.3', 'm811.9', 'm815.8', 'm821.5', 'm835', 'm838.9', 'm846.6', 'm852.4', 'm927.6', 'm977.7', 'm1110.8', 'm1126.2', 'm1282.4', 'm1299.8', 'm1365.4', 'm1411.7', 'm1552.4', 'm1564', 'm1652.7', 'm1810.9', 'm1816.6', 'm1826.3', 'm1835.9', 'm1839.8', 'm1857.1', 'm1862.9', 'm1864.8', 'm1870.6', 'm1878.3', 'm1911.1', 'm1943.9', 'm1947.8', 'm1955.5', 'm1999.8', 'm2019.1', 'm2023', 'm2034.6', 'm2046.1', 'm2065.4', 'm2115.6', 'm2125.2', 'm2165.7', 'm2185', 'm2242.8', 'm2256.3', 'm2264', 'm2312.3', 'm2325.8', 'm2347', 'm2352.8', 'm2372', 'm2985.3', 'm3135.7', 'm3149.2', 'm3372.9', 'm3399.9', 'm3552.3', 'm3625.6', 'm3629.4', 'm3656.4', 'm3673.8', 'm3685.3', 'm3691.1', 'm3696.9', 'm3999.7', 'm4024.8', 'm4040.2', 'm4082.6', 'm4086.5', 'm4094.2', 'm4100', 'm4186.7', 'm4196.4', 'm4202.2', 'm4229.2', 'm4231.1', 'm4244.6', 'm4381.5', 'm4458.7', 'm4476', 'm4499.2', 'm4504.9', 'm4508.8', 'm4562.8', 'm4576.3', 'm4595.6', 'm4647.7', 'm5220.4', 'm5249.3'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m705.8', 'm802.3', 'm1110.8', 'm1282.4', 'm1286.3', 'm1556.3', 'm1999.8', 'm2046.1', 'm2347', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


##15 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture","m4630.3", "m4528.1", "m4510.7", "m4370" , "m4235", "m4192.5", "m3999.7",  "m3685.3", "m3625.6" ,"m3128", "m2987.2" , "m2782.8", "m2713.4", "m2707.6",  "m2672.9",  "m2347", "m2196.5",  "m2044.2", "m2019.1", "m1999.8",  "m1864.8", "m1859.1", "m1843.6", "m1835.9",  "m1683.6", "m1664.3", "m1641.1", "m1618", "m1583.3", "m1488.8", "m1378.9", "m1363.4", "m1284.4", "m1282.4", "m1110.8", "m977.7", "m931.5", "m925.7", "m802.3" ,"m784.9", "m744.4", "m727", "m702", "m694.3", "m607.5"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set
fruits = pd.DataFrame(df, columns= ['Texture','m607.5', 'm613.3', 'm624.8', 'm630.6', 'm634.5', 'm646', 'm665.3', 'm669.2', 'm694.3', 'm702', 'm732.8', 'm740.5', 'm744.4', 'm781', 'm784.9', 'm790.7', 'm794.5', 'm802.3', 'm844.7', 'm856.2', 'm860.1', 'm894.8', 'm927.6', 'm931.5', 'm935.3', 'm966.2', 'm977.7', 'm1112.7', 'm1128.2', 'm1249.7', 'm1261.2', 'm1282.4', 'm1365.4', 'm1375', 'm1380.8', 'm1488.8', 'm1525.4', 'm1548.6', 'm1565.9', 'm1579.4', 'm1587.1', 'm1592.9', 'm1598.7', 'm1619.9', 'm1631.5', 'm1641.1', 'm1645', 'm1656.6', 'm1672', 'm1677.8', 'm1681.6', 'm1699', 'm1805.1', 'm1812.8', 'm1818.6', 'm1824.4', 'm1830.1', 'm1834', 'm1839.8', 'm1841.7', 'm1862.9', 'm1878.3', 'm1913.1', 'm1938.1', 'm1943.9', 'm1947.8', 'm1955.5', 'm1967.1', 'm1992.1', 'm1999.8', 'm2024.9', 'm2038.4', 'm2046.1', 'm2053.8', 'm2071.2', 'm2090.5', 'm2113.6', 'm2117.5', 'm2136.8', 'm2142.6', 'm2186.9', 'm2198.5', 'm2225.5', 'm2244.8', 'm2256.3', 'm2312.3', 'm2329.6', 'm2333.5', 'm2348.9', 'm2667.1', 'm2672.9', 'm2682.5', 'm2707.6', 'm2719.2', 'm2723', 'm2750', 'm2771.2', 'm2779', 'm2782.8', 'm2790.5', 'm2800.2', 'm2819.5', 'm2987.2', 'm3002.7', 'm3137.7', 'm3174.3', 'm3534.9', 'm3538.8', 'm3550.3', 'm3625.6', 'm3631.3', 'm3656.4', 'm3669.9', 'm3675.7', 'm3687.3', 'm3693.1', 'm3698.8', 'm3999.7', 'm4044', 'm4073', 'm4184.8', 'm4192.5', 'm4204.1', 'm4208', 'm4229.2', 'm4242.7', 'm4260', 'm4267.7', 'm4271.6', 'm4343', 'm4362.2', 'm4373.8', 'm4503', 'm4514.6', 'm4528.1', 'm4630.3'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture', 'm694.3', 'm702', 'm802.3', 'm923.7', 'm1110.8', 'm1128.2', 'm1282.4', 'm1284.4', 'm1488.8', 'm1641.1', 'm1685.5', 'm1999.8', 'm2347', 'm2672.9', 'm2707.6', 'm2987.2', 'm3625.6', 'm3999.7', 'm4370', 'm4528.1', 'm4630.3'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set final
fruits = pd.DataFrame(df, columns= ['Texture','m694.3', 'm702', 'm802.3', 'm923.7', 'm1110.8', 'm1128.2', 'm1282.4', 'm1488.8', 'm1641.1', 'm1685.5', 'm1999.8', 'm2347', 'm2672.9', 'm2707.6', 'm2987.2', 'm3625.6', 'm3999.7', 'm4370', 'm4528.1', 'm4630.3'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##20 groups
#All Data
fruits = pd.DataFrame(df, columns= ["Texture", "m4593.7", "m4528.1", "m4302.5" , "m4273.5","m4227.2", "m4186.7", "m4100" ,"m4040.2", "m3999.7","m3695", "m3685.3", "m3671.8" , "m3552.3", "m3625.6", "m3197.4", "m3172.4", "m2989.2", "m2711.5", "m2703.7", "m2669", "m2476.2", "m2113.6", "m2065.4", "m2015.3", "m1999.8", "m1864.8", "m1859.1","m1843.6", "m1835.9", "m1753", "m1751.1", "m1741.4", "m1660.4", "m1637.3","m1614.1", "m1556.3", "m1409.7", "m1363.4", "m1284.4", "m1282.4",  "m1378.9", "m1253.5", "m1128.2", "m1110.8","m966.2", "m919.9", "m896.7",  "m831.2","m821.5" , "m802.3","m736.7", "m723.2",  "m705.8", "m694.3"])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Full set
fruits = pd.DataFrame(df, columns= ['Texture','m628.7', 'm653.8', 'm696.2', 'm703.9', 'm725.1', 'm738.6', 'm742.5', 'm754', 'm786.8', 'm790.7', 'm794.5', 'm802.3', 'm811.9', 'm821.5', 'm825.4', 'm833.1', 'm842.7', 'm846.6', 'm854.3', 'm891', 'm898.7', 'm904.5', 'm921.8', 'm966.2', 'm970', 'm975.8', 'm983.5', 'm997', 'm1110.8', 'm1130.1', 'm1159', 'm1247.7', 'm1267', 'm1282.4', 'm1365.4', 'm1382.7', 'm1409.7', 'm1413.6', 'm1558.2', 'm1612.2', 'm1619.9', 'm1633.4', 'm1641.1', 'm1650.8', 'm1658.5', 'm1664.3', 'm1683.6', 'm1702.9', 'm1735.6', 'm1741.4', 'm1751.1', 'm1814.7', 'm1820.5', 'm1839.8', 'm1843.6', 'm1857.1', 'm1864.8', 'm1870.6', 'm1884.1', 'm1897.6', 'm1901.5', 'm1911.1', 'm1942', 'm1949.7', 'm1955.5', 'm1959.3', 'm1999.8', 'm2009.5', 'm2019.1', 'm2053.8', 'm2071.2', 'm2092.4', 'm2096.3', 'm2115.6', 'm2171.5', 'm2179.2', 'm2185', 'm2291', 'm2298.8', 'm2321.9', 'm2331.5', 'm2347', 'm2374', 'm2383.6', 'm2393.3', 'm2453', 'm2476.2', 'm2480', 'm2578.4', 'm2640.1', 'm2667.1', 'm2678.7', 'm2682.5', 'm2690.2', 'm2701.8', 'm2709.5', 'm2713.4', 'm2725', 'm2730.7', 'm2736.5', 'm2769.3', 'm2794.4', 'm2800.2', 'm2815.6', 'm2825.2', 'm2991.1', 'm2994.9', 'm3021.9', 'm3139.6', 'm3147.3', 'm3160.8', 'm3164.7', 'm3174.3', 'm3180.1', 'm3199.4', 'm3280.4', 'm3529.1', 'm3544.6', 'm3552.3', 'm3583.1', 'm3625.6', 'm3629.4', 'm3656.4', 'm3668', 'm3671.8', 'm3675.7', 'm3679.6', 'm3685.3', 'm3696.9', 'm3712.3', 'm3999.7', 'm4007.4', 'm4036.3', 'm4040.2', 'm4082.6', 'm4092.3', 'm4098', 'm4105.8', 'm4109.6', 'm4119.3', 'm4142.4', 'm4157.8', 'm4177.1', 'm4184.8', 'm4190.6', 'm4206', 'm4227.2', 'm4231.1', 'm4244.6', 'm4273.5', 'm4298.6', 'm4341', 'm4350.7', 'm4373.8', 'm4458.7', 'm4503', 'm4508.8', 'm4528.1', 'm4568.6', 'm4578.2', 'm4593.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m1110.8', 'm1126.2', 'm1282.4', 'm1284.4', 'm1378.9', 'm1409.7', 'm1556.3', 'm1614.1', 'm1741.4', 'm1751.1', 'm1753', 'm1999.8', 'm2015.3', 'm2285.3', 'm2667.1', 'm3197.4', 'm3625.6', 'm3999.7', 'm4273.5', 'm4528.1', 'm4593.7', 'm692.3', 'm694.3', 'm703.9', 'm705.8', 'm802.3', 'm919.9'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#Sub set final
fruits = pd.DataFrame(df, columns= ['Texture','m692.3', 'm694.3', 'm703.9', 'm705.8', 'm802.3', 'm919.9', 'm1110.8', 'm1126.2', 'm1282.4', 'm1378.9', 'm1409.7', 'm1556.3', 'm1614.1', 'm1741.4', 'm1751.1', 'm1999.8', 'm2015.3', 'm2285.3', 'm2667.1', 'm3197.4', 'm3625.6', 'm3999.7', 'm4273.5', 'm4528.1', 'm4593.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

###MIR (>2.5 micrometer)
##No Grouping
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m1110.8', 'm3696.9'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##5 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m856.2', 'm925.7','m1110.8', 'm1999.8', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##10 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m705.8', 'm802.3', 'm1110.8', 'm1282.4', 'm1286.3', 'm1556.3', 'm1999.8', 'm2046.1', 'm2347', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##15 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m694.3', 'm702', 'm802.3', 'm923.7', 'm1110.8', 'm1128.2', 'm1282.4', 'm1488.8', 'm1641.1', 'm1685.5', 'm1999.8', 'm2347', 'm2672.9', 'm2707.6', 'm2987.2', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##20 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','m692.3', 'm694.3', 'm703.9', 'm705.8', 'm802.3', 'm919.9', 'm1110.8', 'm1126.2', 'm1282.4', 'm1378.9', 'm1409.7', 'm1556.3', 'm1614.1', 'm1741.4', 'm1751.1', 'm1999.8', 'm2015.3', 'm2285.3', 'm2667.1', 'm3197.4', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')



###VNIR_MIR common MIR removed
##No Grouping
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1420', 'W1990','m1110.8', 'm3696.9'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##5 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1410', 'W1990','W2460','m856.2', 'm925.7','m1110.8', 'm1999.8', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##10 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1360', 'W1410', 'W1860', 'W1900', 'W2010', 'W2090', 'W2160', 'W2210', 'W2290', 'W2380', 'W2460', 'm705.8', 'm802.3', 'm1110.8', 'm1282.4', 'm1286.3', 'm1556.3', 'm1999.8', 'm2046.1', 'm2347', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##15 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1360', 'W1410', 'W1900', 'W1920', 'W1990', 'W2010', 'W2090', 'W2150', 'W2170', 'W2200', 'W2270', 'W2300', 'W2340', 'W2390', 'W2410', 'W2460', 'm694.3', 'm702', 'm802.3', 'm923.7', 'm1110.8', 'm1128.2', 'm1282.4', 'm1488.8', 'm1641.1', 'm1685.5', 'm1999.8', 'm2347', 'm2672.9', 'm2707.6', 'm2987.2', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##20 groups
#Sub set
fruits = pd.DataFrame(df, columns= ['Texture','W1410', 'W1480', 'W1800', 'W1880', 'W1900', 'W2010', 'W2090', 'W2150', 'W2170', 'W2200', 'W2270', 'W2290', 'W2390', 'W2410', 'W2460', 'm692.3', 'm694.3', 'm703.9', 'm705.8', 'm802.3', 'm919.9', 'm1110.8', 'm1126.2', 'm1282.4', 'm1378.9', 'm1409.7', 'm1556.3', 'm1614.1', 'm1741.4', 'm1751.1', 'm1999.8', 'm2015.3', 'm2285.3', 'm2667.1', 'm3197.4', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')




####Global data FINAL####
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

##VNIR
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df =  pd.read_csv('VNIR.csv')
name = 'Final_Text_VNIR.txt'
#All Bands
group = 'All Bands'
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#1 Group
group = '1 Group'
fruits = pd.DataFrame(df, columns= ['Texture','1420', '1990'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#5 Group
group = '5 Group'
fruits = pd.DataFrame(df, columns= ['Texture','1410', '1990','2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#10 Group
group = '10 Group'
fruits = pd.DataFrame(df, columns= ['Texture','1360', '1410', '1860', '1900', '2010', '2090', '2160', '2210', '2290', '2380', '2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#15 Group
group = '15 Group'
fruits = pd.DataFrame(df, columns= ['Texture','1360', '1410', '1900', '1920', '1990', '2010', '2090', '2150', '2170', '2200', '2270', '2300', '2340', '2390', '2410', '2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#20 Group
group = '20 Group'
fruits = pd.DataFrame(df, columns= ['Texture','1410', '1480', '1800', '1880', '1900', '2010', '2090', '2150', '2170', '2200', '2270', '2290', '2390', '2410', '2460'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##MIR
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df1 =  pd.read_csv('MIR1.csv')
df1 = df1.drop(df1.loc[:, 'm698.1':'m601.7'].columns, axis = 1) 
df = pd.DataFrame(df1)
del df1
name = 'Final_Text_MIR.txt'
#All Bands
group = 'All Bands'
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
#1 Group
group = '1 Group'
fruits = pd.DataFrame(df, columns= ['Texture','m1110.8', 'm3696.9'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##5 groups
group = '5 Group'
fruits = pd.DataFrame(df, columns= ['Texture','m856.2', 'm925.7','m1110.8', 'm1999.8', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##10 groups
group = '10 Group'
fruits = pd.DataFrame(df, columns= ['Texture','m705.8', 'm802.3', 'm1110.8', 'm1282.4', 'm1286.3', 'm1556.3', 'm1999.8', 'm2046.1', 'm2347', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##15 groups
group = '15 Group'
fruits = pd.DataFrame(df, columns= ['Texture', 'm702', 'm802.3', 'm923.7', 'm1110.8', 'm1128.2', 'm1282.4', 'm1488.8', 'm1641.1', 'm1685.5', 'm1999.8', 'm2347', 'm2672.9', 'm2707.6', 'm2987.2', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##20 groups
group = '20 Group'
fruits = pd.DataFrame(df, columns= ['Texture', 'm703.9', 'm705.8', 'm802.3', 'm919.9', 'm1110.8', 'm1126.2', 'm1282.4', 'm1378.9', 'm1409.7', 'm1556.3', 'm1614.1', 'm1741.4', 'm1751.1', 'm1999.8', 'm2015.3', 'm2285.3', 'm2667.1', 'm3197.4', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')

##VNIR_MIR
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df1 =  pd.read_csv('VNIR_MIR1.csv')
df1 = df1.drop(df1.loc[:, 'W350':'W400'].columns, axis = 1) 
df1 = df1.drop(df1.loc[:, 'W2451':'W2500'].columns, axis = 1) 
df1 = df1.drop(df1.loc[:, 'm698.1':'m601.7'].columns, axis = 1) 
df = pd.DataFrame(df1)
del df1

name = 'Final_Text_VNIR_MIR.txt'
#All Bands
group = 'All Bands'
fruits = pd.DataFrame(df)
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##1 Group
group = '1 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1420', 'W1990','m1110.8', 'm3696.9'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##5 groups
group = '5 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1410', 'W1990','W2460','m856.2', 'm925.7','m1110.8', 'm1999.8', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##10 groups
group = '10 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1360', 'W1410', 'W1860', 'W1900', 'W2010', 'W2090', 'W2160', 'W2210', 'W2290', 'W2380', 'W2460', 'm705.8', 'm802.3', 'm1110.8', 'm1282.4', 'm1286.3', 'm1556.3', 'm1999.8', 'm2046.1', 'm2347', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##15 groups
group = '15 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1360', 'W1410', 'W1900', 'W1920', 'W1990', 'W2010', 'W2090', 'W2150', 'W2170', 'W2200', 'W2270', 'W2300', 'W2340', 'W2390', 'W2410', 'W2460', 'm702', 'm802.3', 'm923.7', 'm1110.8', 'm1128.2', 'm1282.4', 'm1488.8', 'm1641.1', 'm1685.5', 'm1999.8', 'm2347', 'm2672.9', 'm2707.6', 'm2987.2', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')
##20 groups
group = '20 Group'
fruits = pd.DataFrame(df, columns= ['Texture','W1410', 'W1480', 'W1800', 'W1880', 'W1900', 'W2010', 'W2090', 'W2150', 'W2170', 'W2200', 'W2270', 'W2290', 'W2390', 'W2410', 'W2460', 'm703.9', 'm705.8', 'm802.3', 'm919.9', 'm1110.8', 'm1126.2', 'm1282.4', 'm1378.9', 'm1409.7', 'm1556.3', 'm1614.1', 'm1741.4', 'm1751.1', 'm1999.8', 'm2015.3', 'm2285.3', 'm2667.1', 'm3197.4', 'm3625.6', 'm3999.7'])
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')




####Global data FINAL for PAPER####
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





####Global data to satellite dataset
pwd
ls
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance/Satellite
sats = glob.glob('*.csv')
##Copied the data files in folder containing classification1.py and ran the code
for sat in sats:
    df =  pd.read_csv(sat)
    fruits = df
    runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


#For resampled BM dataset
fruits = pd.DataFrame(df.drop(['Unnamed: 0', 'Batch_labid', 'Sampleno', 'ISO', 'ID', 'HORI', 'BTOP', 'BBOT', 'SAND', 'SILT', 'CLAY', 'Batch_Labid', 'W350', 'W2500'], axis=1))
df1 = pd.read_excel('BM_10nm_resampled.xlsx')
df1.columns = fruits.columns
fruits = fruits.append(df1, ignore_index=True, sort=False)
#feature_names1 = df1.columns[1:]
#X_test = df1[feature_names1]
#X_test = scaler.fit_transform(X_test)
#y_test = df1['Texture']
runfile('/home/chirag/Documents/HSI/Soil/Classification1.py', wdir='/home/chirag/Documents/HSI/Soil')


#Removing duplicates if any
fruits.shape
a = fruits.T.drop_duplicates().T
a = fruits.T
del rslt_df

#Spectral Resampling using SpectRes
fwhm = pd.read_excel('FWHM.xlsx')
df = pd.read_excel('VNIR_1nm.xlsx')
df2 = pd.read_excel('BM_10nm_resampled.xlsx')
df = pd.read_excel('BM_spectra.xlsx')
df = pd.read_excel('EM.xlsx')
df = pd.read_excel('Lab_Master.xlsx')


old_wave = fwhm['Wavelength_ASD_BM_1'][0:2151,].to_numpy()
new_wave = fwhm['AVIRIS_Centre_Wavelength'][0:424,].to_numpy()
#df1 = df.drop(['Texture'], axis = 1).to_numpy()
#df1 = df.drop(['Sample_ID','NBSS ID','Collector','Sand (%)','Silt (%)','Clay (%)','CBD Fe (%)','Texture'], axis = 1).to_numpy()
df1 = df.drop(df.iloc[:, :13], axis = 1).to_numpy() 
resampled = spectres(new_wave, old_wave, df1)
resampled = pd.DataFrame(resampled)
resampled.columns = new_wave
resampled = pd.concat([df.iloc[:,:13],resampled], axis=1)
#resampled['Texture'] = df['Texture']
#resampled['Sample_ID'] = df['Sample_ID']
#resampled.to_excel('BM_5nm_resampled.xlsx')
resampled.to_excel('Lab_Master_5nm_resampled.xlsx')
df2.to_excel('BM_Global_1nm.xlsx')

## Plotting code:
plt.figure(figsize=(12,6))
# Plot the spectrum at its original sampling
plt.plot(old_wave, df1[2,:], color="blue", label="Original 1 nm")
# Plot the spectrum on the new wavelength grid
new_wave = new_wave[:,1:]
new_wave = np.delete(new_wave,[0,215],0)
plt.plot(new_wave, df2[2,:], color="red", label="Resampled 10 nm")
plt.ylim(0.0, 1.0)
plt.xlabel("Wavelength (nm)", size=16)
plt.ylabel("Reflectance (%)", size=16)
plt.legend(loc=4)
plt.show()


#
df1 = df.drop(['Texture'], axis = 1)

#df1.T.plot()
plt.figure(figsize=(12,6))
#plt.xlim(df1.columns[0],df1.columns[-1])
plt.ylim(0.0, 1.0)
plt.xlabel("Wavelength (nm)", size=16)
plt.ylabel("Reflectance (%)", size=16)
plt.legend(loc='best')
plt.title('Resampled 1 nm')
plt.plot(df1.T)
plt.plot(df2.loc[2,:].T)
plt.show()


####Global Model for Local mapping####
#Dataset Preparation
#df =  pd.read_excel('Aviris_spectra.xlsx')
df =  pd.read_csv('VNIR.csv')
df =  pd.read_excel('VNIR_1nm.xlsx')
df1 =  pd.read_excel('BM_spectra.xlsx')
df = df.drop(['Texture'], axis = 1)
column_list = df.columns.values
column_list1 = column_list[1:217]
column_list2 = [i[1:] for i in column_list1]
column_list3 = ['band_' + i for i in column_list2]
column_list4 = ['Texture']
column_list4.extend(column_list3)
df2 = df1[column_list4]
df2.to_excel('BM_Global_1nm.xlsx')




df1 = pd.DataFrame()
for i in range(215):
    a = column_list[i]
    b = column_list[i+1]
    a1 = df[a]
    b1 = df[b]
    j_list = list(range(1,11))
    for j in j_list: 
        j2 = j/10
        j1 = 1-j2
        df1[str(a) + '_' + str(j)] = j1 * a1 + j2 * b1

df1['W350'] = df['W350']
df1['Texture'] = df['Texture']
df1.to_excel('VNIR_1nm.xlsx')




####Rough####
##Analysing the bands selected from PIC
List1 = '"m4528.1" "m4593.7" "m4273.5" "m4344.9" "m3999.7" "m4042.1" "m4177.1" "m4098"   "m4036.3" "m4244.6" "m3625.6" "m3712.3" "m3668"   "m3679.6" "m3180.1" "m3197.4" "m2994.9" "m2667.1" "m2730.7" "m2815.6" "m2719.2" "m2800.2" "m2480" "m2578.4" "m1999.8" "m2069.3" "m2096.3" "m2053.8" "m1911.1" "m1942"   "m1855.2" "m1753"   "m1751.1" "m1741.4" "m1614.1" "m1631.5" "m1409.7" "m1554.4" "m1363.4" "m1282.4" "m1284.4" "m1380.8" "m1110.8" "m1126.2" "m1255.4" "m919.9"  "m904.5"  "m979.7"  "m821.5"  "m833.1"  "m854.3"  "m802.3"  "m703.9"  "m734.8"  "m754" "m694.3"  "m688.5"  "m692.3"  "m4528.1" "m4593.7" "m4508.8" "m4578.2" "m4273.5" "m3999.7" "m4042.1" "m4192.5" "m4086.5" "m3625.6" "m3683.4" "m3199.4" "m3172.4" "m3195.5" "m2985.3" "m2667.1" "m2709.5" "m2725"   "m2728.8" "m2347"   "m2331.5" "m2383.6" "m2321.9" "m2393.3" "m2298.8" "m1999.8" "m2069.3" "m2015.3" "m2092.4" "m1845.6" "m1837.8" "m1857.1" "m1864.8" "m1753"   "m1751.1" "m1741.4" "m1614.1" "m1633.4" "m1702.9" "m1621.9" "m1409.7" "m1556.3" "m1363.4" "m1282.4" "m1286.3" "m1378.9" "m1110.8" "m1128.2" "m1249.7" "m919.9"  "m896.7"  "m972" "m894.8"  "m802.3"  "m705.8"  "m732.8"  "m725.1"  "m694.3"  "m4528.1" "m4593.7" "m4273.5" "m4343"   "m3999.7" "m4042.1" "m4092.3" "m4208"   "m3625.6" "m3683.4" "m3280.4" "m3160.8" "m3020"   "m3147.3" "m2667.1" "m2725" "m2736.5" "m2690.2" "m2476.2" "m2285.3" "m2289.1" "m1999.8" "m2115.6" "m2015.3" "m2185"   "m2067.3" "m1843.6" "m1837.8" "m1861"   "m1997.9" "m1901.5" "m1947.8" "m1753"   "m1751.1" "m1741.4" "m1614.1" "m1641.1" "m1658.5" "m1409.7" "m1556.3" "m1558.2" "m1361.5" "m1282.4" "m1288.2" "m1378.9" "m1110.8" "m1126.2" "m1253.5" "m919.9" "m896.7"  "m979.7"  "m983.5"  "m804.2"  "m703.9"  "m786.8"  "m806.1"  "m694.3"  "m4528.1" "m4593.7" "m4277.4" "m4298.6" "m3999.7" "m4042.1" "m4184.8" "m4084.5" "m4208"   "m3625.6" "m3685.3" "m3544.6" "m3671.8" "m3693.1" "m3170.4" "m2682.5" "m2701.8" "m2730.7" "m2794.4" "m2478.1" "m2640.1" "m1999.8" "m2069.3" "m2019.1" "m2171.5" "m1909.2" "m1949.7" "m1855.2" "m1959.3" "m1753"   "m1648.9" "m1685.5" "m1664.3" "m1409.7" "m1556.3" "m1359.6" "m1282.4" "m1284.4" "m1378.9" "m1112.7" "m1128.2" "m1247.7" "m919.9"  "m854.3"  "m981.6"  "m970"    "m821.5" "m842.7"  "m802.3"  "m705.8"  "m792.6"  "m811.9"  "m694.3"  "m692.3"  "m690.4"  "m653.8"  "m4528.1" "m4593.7" "m4273.5" "m4341"   "m3999.7" "m4040.2" "m4007.4" "m4082.6" "m4190.6" "m3625.6" "m3683.4" "m3550.3" "m3696.9" "m3675.7" "m3172.4" "m3197.4" "m2991.1" "m2667.1" "m2730.7" "m2721.1" "m2347"   "m2285.3" "m2291"   "m1999.8" "m2069.3" "m2015.3" "m2096.3" "m1845.6" "m1834"   "m1961.3" "m1862.9" "m1907.3" "m1753"   "m1751.1" "m1741.4" "m1650.8" "m1619.9" "m1641.1" "m1409.7" "m1556.3" "m1359.6" "m1282.4" "m1284.4" "m1378.9" "m1110.8" "m1128.2" "m1255.4" "m919.9"  "m894.8"  "m997"    "m848.5"  "m802.3"  "m703.9"  "m792.6"  "m696.2"  "m694.3"  "m628.7" "m686.5"  "m692.3"  "m4528.1" "m4593.7" "m4273.5" "m4302.5" "m3999.7" "m4142.4" "m4159.8" "m4119.3" "m3625.6" "m3683.4" "m3671.8" "m3696.9" "m3172.4" "m3197.4" "m2987.2" "m2667.1" "m2723"   "m2825.2" "m2713.4" "m2474.3" "m2285.3" "m2291"   "m1999.8" "m2113.6" "m2051.9" "m2067.3" "m1820.5" "m1814.7" "m1870.6" "m1897.6" "m1997.9" "m1753"   "m1648.9" "m1683.6" "m1616.1" "m1409.7" "m1554.4" "m1413.6" "m1359.6" "m1282.4" "m1286.3" "m1382.7" "m1284.4" "m1110.8" "m1130.1" "m1249.7" "m921.8"  "m896.7"  "m975.8"  "m821.5"  "m825.4"  "m802.3"  "m705.8" "m742.5"  "m790.7"  "m694.3"  "m4528.1" "m4593.7" "m4503"   "m4568.6" "m4275.5" "m3999.7" "m4040.2" "m4177.1" "m4105.8" "m3625.6" "m3685.3" "m3199.4" "m3583.1" "m3164.7" "m3021.9" "m3139.6" "m2769.3" "m2478.1" "m2285.3" "m2287.2" "m1999.8" "m2113.6" "m2015.3" "m2065.4" "m1847.5" "m1839.8" "m1853.3" "m1884.1" "m1753"   "m1751.1" "m1741.4" "m1614.1" "m1631.5" "m1409.7" "m1556.3" "m1363.4" "m1282.4" "m1284.4" "m1378.9" "m1110.8" "m1128.2" "m1249.7" "m919.9"  "m894.8"  "m970"    "m856.2"  "m802.3"  "m703.9"  "m736.7"  "m756"    "m694.3"  "m4528.1" "m4593.7" "m4275.5" "m4300.5" "m3999.7" "m4142.4" "m4157.8" "m4109.6" "m4227.2" "m3625.6" "m3685.3" "m3199.4" "m3583.1" "m3695"   "m3174.3" "m3020"   "m3147.3" "m2678.7" "m2709.5" "m3149.2" "m2480"   "m2285.3" "m2289.1" "m1999.8" "m2071.2" "m2179.2" "m2009.5" "m1843.6" "m1835.9" "m1861"   "m1870.6" "m1947.8" "m1753"   "m1751.1" "m1741.4" "m1612.2" "m1633.4" "m1409.7" "m1556.3" "m1365.4" "m1282.4" "m1284.4" "m1375"   "m1110.8" "m1126.2" "m1257.4" "m919.9"  "m898.7"  "m966.2"  "m821.5"  "m802.3"  "m705.8"  "m786.8"  "m707.8"  "m694.3"  "m688.5" "m692.3"  "m4528.1" "m4593.7" "m4273.5" "m4458.7" "m4373.8" "m3999.7" "m4040.2" "m4206"   "m4100"   "m3656.4" "m3629.4" "m3681.5" "m3529.1" "m3164.7" "m3197.4" "m2989.2" "m2476.2" "m1999.8" "m2113.6" "m2053.8" "m2015.3" "m1843.6" "m1837.8" "m1862.9" "m1997.9" "m1753"   "m1751.1" "m1735.6" "m1612.2" "m1409.7" "m1556.3" "m1363.4" "m1282.4" "m1284.4" "m1376.9" "m1110.8" "m1126.2" "m1251.6" "m1159"   "m919.9"  "m894.8"  "m977.7"  "m846.6" "m898.7"  "m802.3"  "m705.8"  "m794.5"  "m819.6"  "m694.3"  "m688.5"  "m692.3"  "m4528.1" "m4593.7" "m4273.5" "m4350.7" "m3999.7" "m4142.4" "m4157.8" "m4101.9" "m4231.1" "m3625.6" "m3685.3" "m3552.3" "m3671.8" "m3693.1" "m3174.3" "m3197.4" "m2989.2" "m2667.1" "m2707.6" "m2719.2" "m2678.7" "m2347"   "m2285.3" "m2291"   "m2453" "m2374"   "m1999.8" "m2115.6" "m2017.2" "m2065.4" "m1911.1" "m1945.8" "m1853.3" "m1955.5" "m1753"   "m1751.1" "m1735.6" "m1614.1" "m1409.7" "m1556.3" "m1361.5" "m1282.4" "m1284.4" "m1378.9" "m1110.8" "m1126.2" "m1267" "m919.9"  "m854.3"  "m979.7"  "m891" "m802.3"  "m703.9" "m738.6" "m756" "m694.3"'
List1 = List1.replace(' ', ',')
List1 = List1.replace(',,,', ',')
List1 = List1.replace(',,', ',')
#List = List1.split(' ')
print(List1)
List = ["m4528.1","m4593.7","m4273.5","m4344.9","m3999.7","m4042.1","m4177.1","m4098","m4036.3","m4244.6","m3625.6","m3712.3","m3668","m3679.6","m3180.1","m3197.4","m2994.9","m2667.1","m2730.7","m2815.6","m2719.2","m2800.2","m2480","m2578.4","m1999.8","m2069.3","m2096.3","m2053.8","m1911.1","m1942","m1855.2","m1753","m1751.1","m1741.4","m1614.1","m1631.5","m1409.7","m1554.4","m1363.4","m1282.4","m1284.4","m1380.8","m1110.8","m1126.2","m1255.4","m919.9","m904.5","m979.7","m821.5","m833.1","m854.3","m802.3","m703.9","m734.8","m754","m694.3","m688.5","m692.3","m4528.1","m4593.7","m4508.8","m4578.2","m4273.5","m3999.7","m4042.1","m4192.5","m4086.5","m3625.6","m3683.4","m3199.4","m3172.4","m3195.5","m2985.3","m2667.1","m2709.5","m2725","m2728.8","m2347","m2331.5","m2383.6","m2321.9","m2393.3","m2298.8","m1999.8","m2069.3","m2015.3","m2092.4","m1845.6","m1837.8","m1857.1","m1864.8","m1753","m1751.1","m1741.4","m1614.1","m1633.4","m1702.9","m1621.9","m1409.7","m1556.3","m1363.4","m1282.4","m1286.3","m1378.9","m1110.8","m1128.2","m1249.7","m919.9","m896.7","m972","m894.8","m802.3","m705.8","m732.8","m725.1","m694.3","m4528.1","m4593.7","m4273.5","m4343","m3999.7","m4042.1","m4092.3","m4208","m3625.6","m3683.4","m3280.4","m3160.8","m3020","m3147.3","m2667.1","m2725","m2736.5","m2690.2","m2476.2","m2285.3","m2289.1","m1999.8","m2115.6","m2015.3","m2185","m2067.3","m1843.6","m1837.8","m1861","m1997.9","m1901.5","m1947.8","m1753","m1751.1","m1741.4","m1614.1","m1641.1","m1658.5","m1409.7","m1556.3","m1558.2","m1361.5","m1282.4","m1288.2","m1378.9","m1110.8","m1126.2","m1253.5","m919.9","m896.7","m979.7","m983.5","m804.2","m703.9","m786.8","m806.1","m694.3","m4528.1","m4593.7","m4277.4","m4298.6","m3999.7","m4042.1","m4184.8","m4084.5","m4208","m3625.6","m3685.3","m3544.6","m3671.8","m3693.1","m3170.4","m2682.5","m2701.8","m2730.7","m2794.4","m2478.1","m2640.1","m1999.8","m2069.3","m2019.1","m2171.5","m1909.2","m1949.7","m1855.2","m1959.3","m1753","m1648.9","m1685.5","m1664.3","m1409.7","m1556.3","m1359.6","m1282.4","m1284.4","m1378.9","m1112.7","m1128.2","m1247.7","m919.9","m854.3","m981.6","m970","m821.5","m842.7","m802.3","m705.8","m792.6","m811.9","m694.3","m692.3","m690.4","m653.8","m4528.1","m4593.7","m4273.5","m4341","m3999.7","m4040.2","m4007.4","m4082.6","m4190.6","m3625.6","m3683.4","m3550.3","m3696.9","m3675.7","m3172.4","m3197.4","m2991.1","m2667.1","m2730.7","m2721.1","m2347","m2285.3","m2291","m1999.8","m2069.3","m2015.3","m2096.3","m1845.6","m1834","m1961.3","m1862.9","m1907.3","m1753","m1751.1","m1741.4","m1650.8","m1619.9","m1641.1","m1409.7","m1556.3","m1359.6","m1282.4","m1284.4","m1378.9","m1110.8","m1128.2","m1255.4","m919.9","m894.8","m997","m848.5","m802.3","m703.9","m792.6","m696.2","m694.3","m628.7","m686.5","m692.3","m4528.1","m4593.7","m4273.5","m4302.5","m3999.7","m4142.4","m4159.8","m4119.3","m3625.6","m3683.4","m3671.8","m3696.9","m3172.4","m3197.4","m2987.2","m2667.1","m2723","m2825.2","m2713.4","m2474.3","m2285.3","m2291","m1999.8","m2113.6","m2051.9","m2067.3","m1820.5","m1814.7","m1870.6","m1897.6","m1997.9","m1753","m1648.9","m1683.6","m1616.1","m1409.7","m1554.4","m1413.6","m1359.6","m1282.4","m1286.3","m1382.7","m1284.4","m1110.8","m1130.1","m1249.7","m921.8","m896.7","m975.8","m821.5","m825.4","m802.3","m705.8","m742.5","m790.7","m694.3","m4528.1","m4593.7","m4503","m4568.6","m4275.5","m3999.7","m4040.2","m4177.1","m4105.8","m3625.6","m3685.3","m3199.4","m3583.1","m3164.7","m3021.9","m3139.6","m2769.3","m2478.1","m2285.3","m2287.2","m1999.8","m2113.6","m2015.3","m2065.4","m1847.5","m1839.8","m1853.3","m1884.1","m1753","m1751.1","m1741.4","m1614.1","m1631.5","m1409.7","m1556.3","m1363.4","m1282.4","m1284.4","m1378.9","m1110.8","m1128.2","m1249.7","m919.9","m894.8","m970","m856.2","m802.3","m703.9","m736.7","m756","m694.3","m4528.1","m4593.7","m4275.5","m4300.5","m3999.7","m4142.4","m4157.8","m4109.6","m4227.2","m3625.6","m3685.3","m3199.4","m3583.1","m3695","m3174.3","m3020","m3147.3","m2678.7","m2709.5","m3149.2","m2480","m2285.3","m2289.1","m1999.8","m2071.2","m2179.2","m2009.5","m1843.6","m1835.9","m1861","m1870.6","m1947.8","m1753","m1751.1","m1741.4","m1612.2","m1633.4","m1409.7","m1556.3","m1365.4","m1282.4","m1284.4","m1375","m1110.8","m1126.2","m1257.4","m919.9","m898.7","m966.2","m821.5","m802.3","m705.8","m786.8","m707.8","m694.3","m688.5","m692.3","m4528.1","m4593.7","m4273.5","m4458.7","m4373.8","m3999.7","m4040.2","m4206","m4100","m3656.4","m3629.4","m3681.5","m3529.1","m3164.7","m3197.4","m2989.2","m2476.2","m1999.8","m2113.6","m2053.8","m2015.3","m1843.6","m1837.8","m1862.9","m1997.9","m1753","m1751.1","m1735.6","m1612.2","m1409.7","m1556.3","m1363.4","m1282.4","m1284.4","m1376.9","m1110.8","m1126.2","m1251.6","m1159","m919.9","m894.8","m977.7","m846.6","m898.7","m802.3","m705.8","m794.5","m819.6","m694.3","m688.5","m692.3","m4528.1","m4593.7","m4273.5","m4350.7","m3999.7","m4142.4","m4157.8","m4101.9","m4231.1","m3625.6","m3685.3","m3552.3","m3671.8","m3693.1","m3174.3","m3197.4","m2989.2","m2667.1","m2707.6","m2719.2","m2678.7","m2347","m2285.3","m2291","m2453","m2374","m1999.8","m2115.6","m2017.2","m2065.4","m1911.1","m1945.8","m1853.3","m1955.5","m1753","m1751.1","m1735.6","m1614.1","m1409.7","m1556.3","m1361.5","m1282.4","m1284.4","m1378.9","m1110.8","m1126.2","m1267","m919.9","m854.3","m979.7","m891","m802.3","m703.9","m738.6","m756","m694.3"]
print(List)
print(len(List))
fullset = set(List)
print(sorted(fullset))
print(len(fullset))
print(collections.Counter(List))
subset = [item for item, count in collections.Counter(List).items() if count > 4]
print(sorted(set(subset)))
print(len(subset))


##Printing to txt files
stdout_fileinfo = sys.stdout
sys.stdout = open('Test.txt','a')
print('Hello World')
print('Hello World')
sys.stdout.close()
sys.stdout = stdout_fileinfo







####Regression approaches####
#Using R packages in Python
import rpy2
import rpy2.tests
from rpy2.robjects import r,pandas2ri
pandas2ri.activate()
r.data('iris')
r['iris'].head()

import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

#Importing default R pakages in R
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import soiltexture
utils = importr("utils")
utils.data
#Installing R packages not by default in R
packnames = ('ggplot2', 'soiltexture')
from rpy2.robjects.vectors import StrVector
utils.install_packages(StrVector(packnames))

#Importing and using the soil texture package
soiltex = importr("soiltexture")
#Check names of the exported package in R
soiltex._exported_names

soiltex.TT_plot(class_sys = "HYPRES.TT")
df =  pd.read_csv('ASD_Spectra_Master.csv')
df1 = df[['SAND', 'SILT', 'CLAY']]
df2 = soiltex.TT_normalise_sum(tri_data = df1)
a = soiltex.TT_points_in_classes(tri_data = df2, class_sys = "USDA.TT", PiC_type = "1", collapse = ";")
a = soiltex.TT_points_in_classes(tri_data = df2, class_sys = "USDA.TT", PiC_type = "t", collapse = ";")
a


#Soil Texture using Python:
#USDA Texture Triangle Classification:
CLAY1 = df['CLAY'].to_numpy()[1:10]
SILT1 = df['SILT'].to_numpy()[1:10]
SAND1 = df['SAND'].to_numpy()[1:10]

conditions = [
((100 >= CLAY1 >= 40) and (40 >= SILT1 >= 0) and (45 >= SAND1 >= 0)),
((10 >= CLAY1 >= 0) and (15 >= SILT1 >= 0) and (100 >= SAND1 >= 85)),
((12.5 >= CLAY1 >= 0)	and (100 >= SILT1 >= 80) and (20 >= SAND1 >= 0)),
((15 >= CLAY1 >= 0) and (30 >= SILT1 >= 0) and (90	>= SAND1 >= 70)),
((20 >= CLAY1 >= 0) and (50 >= SILT1 >= 0) and (85 >= SAND1 >= 42.50)),
((35 >= CLAY1 >= 20) and (27.5>= SILT1 >= 0) and (80 >= SAND1 >= 45)),
((55 >= CLAY1 >= 35) and (20 >= SILT1 >= 0) and (65 >= SAND1 >= 45)),
((60 >= CLAY1 >= 40) and (60 >= SILT1 >= 40) and (20 >= SAND1 >= 0)),
((40 >= CLAY1 >= 27.5) and (72.5 >= SILT1 >= 40)	and (20 >= SAND1 >= 0)),
((27.5 >= CLAY1 >= 0)	and (87.5 >= SILT1 >= 50) and (50 >= SAND1 >= 0)),
((40 >= CLAY1 >= 27.5) and (52.5 >= SILT1 >= 15) and (45 >= SAND1 >= 20)),
((27.5 >= CLAY1 >= 7.5) and (50 >= SILT1 >= 27.5) and (52.5 >= SAND1 >= 22.5))
]

values = ['Cl', 'Sa', 'Si', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'SiCl', 'SiClLo', 'SiLo','ClLo', 'Lo']

df['Texture1'] = np.select(conditions, values)

Texture1 = [None] * 3643
for i in range(3643):
    if ((100 >= df['CLAY'][i] >= 40) and (40 >= df['SILT'][i] >= 0) and (45 >= df['SAND'][i] >= 0)):
        Texture1[i]  = 'Cl'
    elif ((10 >= df['CLAY'][i] >= 0) and (15 >= df['SILT'][i] >= 0) and (100 >= df['SAND'][i] >= 85)):
        Texture1[i]  = 'Sa'
    elif ((12.5 >= df['CLAY'][i] >= 0)	and (100 >= df['SILT'][i] >= 80) and (20 >= df['SAND'][i] >= 0)):
        Texture1[i]  = 'Si'
    elif ((15 >= df['CLAY'][i] >= 0) and (30 >= df['SILT'][i] >= 0) and (90	>= df['SAND'][i] >= 70)):
        Texture1[i]  = 'LoSa'
    elif ((20 >= df['CLAY'][i] >= 0) and (50 >= df['SILT'][i] >= 0) and (85 >= df['SAND'][i] >= 42.50)):
        Texture1[i]  = 'SaLo'
    elif ((35 >= df['CLAY'][i] >= 20) and (27.5>= df['SILT'][i] >= 0) and (80 >= df['SAND'][i] >= 45)):
        Texture1[i]  = 'SaClLo'
    elif ((55 >= df['CLAY'][i] >= 35) and (20 >= df['SILT'][i] >= 0) and (65 >= df['SAND'][i] >= 45)):
        Texture1[i]  = 'SaCl'
    elif ((60 >= df['CLAY'][i] >= 40) and (60 >= df['SILT'][i] >= 40) and (20 >= df['SAND'][i] >= 0)):
        Texture1[i]  = 'SiCl'
    elif ((40 >= df['CLAY'][i] >= 27.5) and (72.5 >= df['SILT'][i] >= 40)	and (20 >= df['SAND'][i] >= 0)):
        Texture1[i]  = 'SiClLo'
    elif ((27.5 >= df['CLAY'][i] >= 0)	and (87.5 >= df['SILT'][i] >= 50) and (50 >= df['SAND'][i] >= 0)):
        Texture1[i]  = 'SiLo'
    elif ((40 >= df['CLAY'][i] >= 27.5) and (52.5 >= df['SILT'][i] >= 15) and (45 >= df['SAND'][i] >= 20)):
        Texture1[i]  = 'ClLo'
    else:
        Texture1[i]  = 'Lo'

Texture1.unique()
fullset = set(Texture1)
print(len(fullset))
print(collections.Counter(Texture1))
df['Texture1'] = Texture1
df['Check'] = (df['Texture'] == df['Texture1'])
a = df['Check']
print(collections.Counter(a))

print(collections.Counter(sorted(Texture1)))
print(collections.Counter(sorted(df['Texture'])))

for i in range(3643):
    

for i in range(9):
    print(i)
    if ((100 >= CLAY1[i] >= 40) & (40 >= SILT1[i] >= 0) & (45 >= SAND1[i] >= 0)): Texture1[i] = 'Cl'
    if ((20 >= CLAY1[i] >= 0) & (50 >= SILT1[i] >= 0) & (85 >= SAND1[i] >= 42.50)): Texture1[i] = 'SaLo'





####Histogram####
Texture_class = df['Texture']
Texture_class.unique()
len(Texture_class.unique())
fullset = set(Texture_class)
print(len(fullset))
Texture_class1 = collections.Counter(sorted(Texture_class))
df1 = pd.DataFrame.from_dict(Texture_class1, orient='index')
df1.plot(kind='bar')
df1.plot(kind='area')
df1.to_csv('Histogram_Texture')



####Plots####
cd /home/chirag/Documents/HSI/Soil/Bands/Reflectance
df =  pd.read_csv('VNIR1.csv')
df =  pd.read_csv('MIR1.csv')
df =  pd.read_csv('VNIR_MIR.csv')
df =  pd.read_csv('VNIR_MIR_common_rm_MIR.csv')
df =  pd.read_csv('VNIR_MIR_common_rm_VNIR.csv')

cd /home/chirag/Documents/HSI/Soil/Beginning
fwhm = pd.read_excel('FWHM.xlsx')
old_wave = fwhm['Wavelength_ASD_BM_1'][0:2151,].to_numpy()
old_wave = df.columns[1:3577]

df_plot = df.drop(['Texture'], axis = 1).to_numpy()
plt.figure(figsize=(12,6))
plt.plot(old_wave,df_plot[1,:])
plt.ylim(0.0, 1.0)
plt.xlabel("Wavelength (nm)", size=16)
plt.ylabel("Reflectance (%)", size=16)
#plt.legend(loc=4)
#plt.legend(loc='best')
plt.title("Soil spectra in VNIR region", size=18)
plt.show()
    

for i in range(2151):
    plt.plot(old_wave,df_plot[i,:])


from matplotlib.patches import Rectangle
fig, ax = plt.subplots()
plt.ylim(0.0, 0.6)
plt.xlabel("Wavelength (nm)", size=16)
plt.ylabel("Reflectance (%)", size=16)
plt.title("Soil spectra with groups in VNIR region", size=18)
ax.plot(old_wave,df_plot[14,:],label='Cl')
ax.plot(old_wave,df_plot[30,:],label='ClLo')
ax.plot(old_wave,df_plot[880,:],label='Lo')
ax.plot(old_wave,df_plot[1201,:],label='LoSa')
ax.plot(old_wave,df_plot[298,:],label='Sa')
ax.plot(old_wave,df_plot[1816,:],label='SaCl')
ax.plot(old_wave,df_plot[133,:],label='SaClLo')
ax.plot(old_wave,df_plot[119,:],label='SaLo')
ax.plot(old_wave,df_plot[2767,:],label='Si')
ax.plot(old_wave,df_plot[1645,:],label='SiCl')
ax.plot(old_wave,df_plot[402,:],label='SiClLo')
ax.plot(old_wave,df_plot[3640,:],label='SiLo')
ax.annotate(xy=[300,df_plot[14,0]], s='Cl')
ax.annotate(xy=[2500,df_plot[30,2150]], s='ClLo')
ax.annotate(xy=[2500,df_plot[880,2150]], s='Lo')
ax.annotate(xy=[2500,df_plot[1201,2150]], s='LoSa')
ax.annotate(xy=[300,df_plot[298,0]], s='Sa')
ax.annotate(xy=[2500,df_plot[1816,2150]], s='SaCl')
ax.annotate(xy=[2500,df_plot[133,2150]], s='SaClLo')
ax.annotate(xy=[200,df_plot[119,0]], s='SaLo')
ax.annotate(xy=[2500,df_plot[2767,2150]], s='Si')
ax.annotate(xy=[2500,df_plot[1645,2150]], s='SiCl')
ax.annotate(xy=[300,df_plot[402,0]], s='SiClLo')
ax.annotate(xy=[2500,df_plot[3640,2150]], s='SiLo')
ax.add_patch(Rectangle((350,0),150,1,color="blue"))
ax.add_patch(Rectangle((500,0),300,1,color="cyan"))
ax.add_patch(Rectangle((800,0),500,1,color="pink"))
ax.add_patch(Rectangle((1300,0),550,1,color="yellow"))
ax.add_patch(Rectangle((1850,0),650,1,color="orange"))
ax.legend()
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=6)
plt.show()

for i in range(2151):
    ax.plot(old_wave,df_plot[i,:])




#Average spectra for each texture
values = ['Cl', 'Sa', 'Si', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'SiCl', 'SiClLo', 'SiLo','ClLo', 'Lo']
    
mean_df = np.empty([12,3914])
for i_num,i in enumerate(values):
    df1 = df[df['Texture'] == i]
    df1 = df1.drop(['Texture'], axis=1)
    mean_df[i_num,:] = np.mean(df1.to_numpy(), axis = 0)

mean_df = pd.DataFrame(mean_df)
mean_df.columns = df.columns[1:]
mean_df['Texture'] = values
mean_df.to_csv('Mean_spectra.csv')


