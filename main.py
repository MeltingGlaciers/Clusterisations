import pandas as pd
import numpy as np

from sklearn.manifold  import MDS
import ierarchy
import kmeans

def mod_file():
    file = open("D:\\Progs\\ISIS\\Lab1\\binary.csv", "r")
    new_file = open("D:\\Progs\\ISIS\\Lab1\\data.csv", "w")
    for line in file:
        new_line = line.replace(";", ",")
        new_file.write(new_line)

def mds_method(df):
    mds = MDS(n_components=2)
    mds.fit(df)
    return mds

def __main__(filepath="D:\\Progs\\ISIS\Lab1\\binary.dat"):
    pure_data = pd.read_csv(filepath, sep='\t', header=None)
    labels = []
    data = []
    for i in range(pure_data.size):
        temp = pure_data.values[i][0].split()
        labels.append(temp[0])
        data.append(temp[1:])
    df = np.array(data)
    #ierarchy.run(df,mds_method(df))
    kmeans.run(df,mds_method(df))

__main__()