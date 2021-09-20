import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold  import MDS
import matplotlib.pyplot as plt

def elbow_plot(df):
    distortions = []
    K = range(1,10)
    for k in K:
        model = KMeans(n_clusters=k)
        model.fit(df)
        distortions.append(model.inertia_)
    plt.plot(K, distortions, 'bx-')
    plt.show()

def run(df,mds):

    elbow_plot(df)
    model = KMeans(n_clusters=3)
    model.fit(df)
    plt.scatter(mds.embedding_[:,0],mds.embedding_[:,1],c=model.labels_)
    plt.show()
    elbow_plot(df)