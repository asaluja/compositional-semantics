#!/usr/bin/python -tt

'''
File: learning.py
Date: July 2, 2014
Description: this script takes as a list of word pairs, as well as the
vector representations of those words, and sets up a regression problem
to learn the parameters in our composition function. 
Usage: 
'''

import sys, commands, string, getopt
import numpy as np
import sklearn.linear_model as regressor

def readVecFile(filename, normalize):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep)) if normalize else rep
    return repDict

def createTraining(wordVecs):
    training_tuples = []
    counter = 0
    for line in sys.stdin:
        counter += 1
        elements = line.strip().split(' ||| ')
        output = elements[0]
        input_left = elements[1].split()[0]
        input_right = elements[1].split()[1]
        if output in wordVecs and input_left in wordVecs and input_right in wordVecs: #may want to print out some information on the 'else' condition
            training_tuples.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right]))
    print "Out of %d training examples, input-output triples exist for %d training examples"%(counter, len(training_tuples))
    return training_tuples

def learnParameters(training_data):
    numSamples = len(training_data)
    dim = len(training_data[0][0])
    P = dim * dim
    print "Number of training examples: %d; Number of regression problems: %d; Number of covariates: %d"%(numSamples, dim, P)
    y = np.zeros((numSamples, dim))
    X = np.zeros((numSamples, P))
    for idx, triple in enumerate(training_data): 
        y[idx,:] = triple[0].transpose()
        X[idx,:] = np.hstack(np.outer(triple[1], triple[2]))
    print "Completed assembling training data into regression format.  Now starting regression."
    #print "sample row: "
    #print X[0,:]
    print "Sample output: "
    print y[:,0]
    #lasso = regressor.Lasso(alpha=alpha) #NOTE the max_iter parameter
    for col in xrange(y.shape[1]): #loop through all columns
        #lasso = regressor.LassoLars(alpha=0.00000001)
        lasso = regressor.LassoLarsCV(verbose=True, n_jobs=16)
        lasso.fit(X, y[:,col])
        print "Maximum coefficient value: %.3g"%(max(lasso.coef_))
        print "Intercept value: %.3g"%(lasso.intercept_)

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'n')
    normalize = False
    for opt in opts:
        if opt[0] == '-n':
            normalize = True
    wordVecs = readVecFile(args[0], normalize)
    training_data = createTraining(wordVecs)
    learnParameters(training_data)

if __name__ == "__main__":
    main()
