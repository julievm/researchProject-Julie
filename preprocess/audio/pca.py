import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# used to replace time frame into a format that can be converted to float (1.000.000 -> 1.000000)
def fix_float(s):
    return s.replace('.', '[DOT]', 1).replace('.', '').replace('[DOT]', '.')

#takes in a filename and returns the standardized features
def standardizeFeatures(fileName):
    # get data from csv file
    dataset = pd.read_csv('audio_nopca\\'+fileName, sep=";")

    # get data in right format to go from string to float
    dataset["frameTime"] = [fix_float(s) for s in dataset["frameTime"]]
    dataset = dataset.replace(',', '.', regex=True)

    # remove the frameTime column from the data
    dataset2 = dataset
    dataset_notime = dataset2.loc[:, dataset2.columns != "frameTime"]
    print(dataset_notime)

    # standardize the features
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(dataset_notime))
    return scaled_data


# take in a csv file, perform pca then return the new csv
def pcsTry():
    #get standardized features of all csv files
    data2 = standardizeFeatures('2noPCA.csv')
    
    #apply PCA
    pca = PCA(n_components= 6)
    
    #only fit once to apply same pca on all participants
    pca.fit(data2)

    # dir = 'C:\\Users\\waded\\Documents\\opensmile-3.0.1-win-x64\\bin\\wavs\\'
    dir = '' #directory with all participant csvs before pca
    for f in os.listdir(dir):#directory with all participant csvs
        data = standardizeFeatures(f)
        #apply dimentional reduction for all participants
        data_pca = pca.transform(data)
        data_pca = pd.DataFrame(data_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"])
        name = f.split("_")[0]
        #create new csv's
        data_pca.to_csv(name, index=False)


pcsTry()