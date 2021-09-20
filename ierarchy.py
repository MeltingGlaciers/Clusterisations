import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def plot_dendrogram(model, **kwargs):
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

    dendrogram(linkage_matrix, **kwargs)

def run(df,mds):
    model = AgglomerativeClustering(distance_threshold=3, n_clusters=None,compute_distances=True)

    model = model.fit(df)
    plot_dendrogram(model, truncate_mode='level', p=16)
    plt.show()
    plt.scatter(mds.embedding_[:, 0], mds.embedding_[:, 1], c=model.labels_)
    plt.show()

