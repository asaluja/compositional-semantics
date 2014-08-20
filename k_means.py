#!/usr/bin/python -tt

'''
File: k_means.py
Date:
Description: does k-means clustering on word vectors using scikit learn
'''

import sys, commands, string
import numpy as np
import sklearn.cluster as cluster

def main():
    IDtoWord = {}
    raw_data = []
    ID = 0
    numClusters = int(sys.argv[1])
    for line in sys.stdin:
        elements = line.strip().split()
        if len(elements) > 2:
            word = elements[0]
            IDtoWord[ID] = word
            ID += 1
            raw_data.append(map(float, elements[1:]))
    X = np.array(raw_data)
    km = cluster.MiniBatchKMeans(n_clusters=numClusters, init='k-means++')
    result = km.fit_predict(X)
    sil = sklearn.metrics.silhouette_score(X, result.labels_, metric='euclidean')
    sys.stderr.write("Silhouette score: %.3f\n"%sil)
    clusters = {}
    for idx, clusterID in enumerate(result):
        word = IDtoWord[idx]
        words = [] if clusterID not in clusters else clusters[clusterID]
        words.append(word)
        clusters[clusterID] = words
    for clusterID in clusters:
        #print "%d:"%clusterID,
        for word in clusters[clusterID]:
            print "%d:%s"%(clusterID, word)

if __name__ == "__main__":
    main()
