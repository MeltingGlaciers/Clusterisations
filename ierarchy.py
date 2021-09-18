import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def __main__(filepath="D:\\Progs\\ISIS\Lab1\\binary.dat"):
    pure_data = pd.read_csv(filepath, sep='\t',header=None)
    data = []
    for i in range(pure_data.size):
        col = pure_data.values[i][0].split()[1:]
        data.append(col)

    df = np.array(data)
    clus = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
   # labels = clus.fit_predict(df)

   # print(labels)

    #plt.scatter(df[:, 0], df[:, 1], c=labels)
    #plt.show()
    #dendrogram(clus,truncate_mode='level',p=3)
    #plt.show()

    iris = load_iris()
    X = iris.data

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(df)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=4)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

__main__()

