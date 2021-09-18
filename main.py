import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def test():
    data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6,
                      random_state=50)  # create np array for data points
    points = data[0]  # create scatter plot
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='viridis')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()
#test()

def mod_file():
    file = open("D:\\Progs\\ISIS\\Lab1\\binary.csv", "r")
    new_file = open("D:\\Progs\\ISIS\\Lab1\\data.csv", "w")
    for line in file:
        new_line = line.replace(";", ",")
        new_file.write(new_line)

def __main__(filepath="D:\\Progs\\ISIS\Lab1\\binary.dat"):
    pure_data = pd.read_csv(filepath, sep='\t',header=None)
    data = []
    for i in range(pure_data.size):
        col = pure_data.values[i][0].split()[1:]
        data.append(col)

    df = np.array(data)




__main__()