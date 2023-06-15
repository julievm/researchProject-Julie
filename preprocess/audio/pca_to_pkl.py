import numpy as np
import os
import pickle as pkl
# participants




dict = {}

#dir = 'C:\\Users\\waded\\Documents\\opensmile-3.0.1-win-x64\\bin\\afterPCA\\'
dir = '' #directory with all csv files after PCA
for f in os.listdir(dir):#directory with all participant csvs
    csv = open(dir + f)
    data = np.loadtxt(csv, delimiter=',', skiprows=1)


    key = f.split('.')[0]
    
    n = 100
    final = [data[i* n:(i+1) * n ] for i in range((len(data) + n-1) // n)]

    dict[key] = final


with open('audio.pkl', 'wb') as op: 
    pkl.dump(dict, op)



