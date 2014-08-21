#!/usr/bin/python -tt

'''
File: k_means.py
Date: August 20, 2014
Description: does k-means clustering on word vectors using scikit learn
'''

import sys, commands, string, getopt
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as metrics

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'b')
    bigram = False
    for opt in opts:
        if opt[0] == '-b': #bigram
            bigram = True
    numClusters = int(args[0])
    IDtoWord = {}
    raw_data = []
    ID = 0
    for line in sys.stdin:
        elements = line.strip().split()
        if len(elements) > 2:
            word = ' '.join(elements[:2]) if bigram else elements[0]
            IDtoWord[ID] = word
            ID += 1
            if bigram:
                raw_data.append(map(float, elements[2:]))
            else:
                raw_data.append(map(float, elements[1:]))
    X = np.array(raw_data)
    km = cluster.MiniBatchKMeans(n_clusters=numClusters, init='k-means++')
    result = km.fit_predict(X)
    sil = metrics.silhouette_score(X, result, metric='euclidean', sample_size=1000)
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
