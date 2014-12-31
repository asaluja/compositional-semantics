#!/usr/bin/python -tt

'''
File: learning.py
Date: July 2, 2014
Description: this script takes as a list of word pairs, as well as the
vector representations of those words, and sets up a regression problem
to learn the parameters in our composition function. 
Usage: python learning.py wordVectorsIn ParametersOut < training_data
Update (August 22, 2014): modified the script to take into account modified
handling of PPDB training extraction. 
'''

import sys, commands, string, getopt, cPickle, math
import numpy as np
import multiprocessing as mp
import sklearn.linear_model as regressor
from extract_training import *

'''
read in word representations from text file
'''
def readVecFile(filename, normalize):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep)) if normalize else rep
    return repDict

'''
from PPDB, add the training example if it meets certain criteria
'''
def addTrainingExample(wordVecs, filterDup, training_data, training_stats, key, output, input_left, input_right):
    training_pos = training_data[key] if key in training_data else []
    num_train = training_stats[key] if key in training_stats else [0,0]
    num_train[0] += 1
    if output in wordVecs and input_left in wordVecs and input_right in wordVecs: 
        if filterDup:
            if input_left != output and input_right != output:
                training_pos.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right]))
                num_train[1] += 1
            else:
                sys.stderr.write("Filtered example %s ||| %s because of redundancy\n"%(output, ' '.join([input_left, input_right])))
        else:
            training_pos.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right]))
            num_train[1] += 1
        training_data[key] = training_pos
    else:
        sys.stderr.write("Could not find one of the following words in the vocabulary: %s, %s, or %s\n"%(output, input_left, input_right))

'''
function that goes through PPDB and assembles the training data by calling addTrainingExample and doing some pre and post-processing
'''
def createTraining(dbloc, wordVecs, filterDup):
    print "Extracting training examples directly from PPDB"
    extractor = TrainingExtractor(dbloc, "all")
    extractor.extract_examples()
    training_tuples = extractor.return_examples()
    training_data = {}
    training_stats = {}
    for key, one_phrase, many_phrase in training_tuples:
        input_left = many_phrase.split()[0]
        input_right = many_phrase.split()[1]
        addTrainingExample(wordVecs, filterDup, training_data, training_stats, key, one_phrase, input_left, input_right)
    for key in training_stats:
        print "POS Pair %s: out of %d training examples, valid input-output triples exist for %d examples"%(key, training_stats[key][0], training_stats[key][1])
    return training_data

def readTrainingFromFile(trainFile, wordVecs, filterDup):
    print "Reading training examples from file"
    training_data = {}
    training_stats = {}
    train_fh = open(trainFile, 'rb')
    for line in train_fh:
        elements = line.strip().split(' ||| ')
        key = elements[0]
        for triple in elements[1:]:
            assert len(triple.split()) == 3
            one_phrase, input_left, input_right = triple.split()
            addTrainingExample(wordVecs, filterDup, training_data, training_stats, key, one_phrase, input_left, input_right)
        print "Completed reading in examples for %s"%key
    train_fh.close()
    return training_data

'''
distributed implementation of multivariate regression by doing the elements/coordinates independently.
There is something to be lost by this in that R^2 is always less than the multivariate solution,
but this difference is usually minimal and we can take advantage of multiple cores.  Only works
for particular types of regressions: lasso, ridge, lars, and elastic
'''
def regressorParallel(data, labels, start, end, regStr, out_q):
    reg = None
    if regStr == "lasso":
        reg = regressor.LassoCV()
    elif regStr == "ridge":
        reg = regressor.RidgeCV()
    elif regStr == "lars":
        reg = regressor.LassoLarsCV(n_jobs=1)
    elif regStr == "elastic":
        reg = regressor.ElasticNetCV()
    coefficients = []
    for idx in range(start, end):        
        reg.fit(data, labels[:,idx]) #what would happen if we passed in labels, not labels[:,idx]? 
        print "Dimension %d Alpha selected: %.3g"%(idx, reg.alpha_) #use this for CV experiments
        R2 = reg.score(data, labels[:,idx])
        print "Dimension %d R^2 on data: %.3f"%(idx, R2)
        print "Dimension %d Number of non-zero values in coefficients: %d"%(idx, (reg.coef_ != 0).sum())
        coefficients.append((idx, R2, reg.coef_, reg.intercept_))
    out_q.put(coefficients)

'''
standard multivariate regression where it is done together
'''
def regressorMultivariate(data, labels, regStr):
    reg = None
    if regStr == "lasso":
        reg = regressor.LassoCV()
    elif regStr == "ridge":
        reg = regressor.RidgeCV()
    elif regStr == "lars":
        reg = regressor.LassoLarsCV(n_jobs=1)
    elif regStr == "elastic":
        reg = regressor.ElasticNetCV()
    reg.fit(data, labels)
    print "Multivariate Alpha selected: %.3g"%reg.alpha_
    R2 = reg.score(data, labels)
    print "Multivariate R^2 on data: %.3f"%R2
    print "Number of non-zero values in coefficients: %d"%((reg.coef_ != 0).sum())
    return (reg.coef_, reg.intercept_)

'''
experimental regression function where the prior is linguistically motivated, so
we impose some kind of structural sparsity to the parameter matrix. 
Only works for concatenative models
'''
def regressorLinguisticPrior(X, y, pos_pair, alpha, dim):
    numExamples = X.shape[0]
    print pos_pair #make change: only do this for certain combinations
    X = np.concatenate((np.ones((numExamples,1)), X), axis=1)
    left_mat = np.zeros((dim, dim+1)) #+1 because of intercept
    right_mat = np.identity(dim)*alpha
    W_star = np.concatenate((left_mat, right_mat), axis=1) #vx(p+1) matrix
    #set b depending on what the POS pair is, otherwise can just always set it to the 'else' value
    b = np.dot(X.T, y) if pos_pair == "X X" or pos_pair == "NN NN" else np.dot(X.T, y) + W_star.T #pxv matrix
    A = np.dot(X.T, X) + alpha*np.identity(W_star.shape[1]) #(p+1)x(p+1) matrix
    W = np.linalg.solve(A, b) #result should be (p+1)xv matrix
    intercept = W[0,:]
    return (W[1:,:].T, intercept)

'''
wrapper function for learning parameters
'''
def learnParameters(training_data, pos_pair, numProc, diagonal, concat, reg, multivariate, alpha):
    numSamples = len(training_data)
    dim = len(training_data[0][0])
    P = dim if diagonal else dim * dim
    if concat: 
        P = 2*dim
    print "Number of training examples: %d; Number of regression problems: %d; Number of covariates: %d"%(numSamples, dim, P)
    y = np.zeros((numSamples, dim))
    X = np.zeros((numSamples, P))
    for idx, triple in enumerate(training_data): #assemble the data in y and X
        y[idx,:] = triple[0].transpose()
        if concat:
            X[idx,:] = np.concatenate((triple[1], triple[2]), axis=1)
        elif diagonal:
            X[idx,:] = np.diagonal(np.outer(triple[1], triple[2]))
        else:
            X[idx,:] = np.hstack(np.outer(triple[1], triple[2]))
    print "Completed assembling training data into regression format.  Now starting regression."
    parameter = np.zeros((dim, dim, dim)) if not concat else np.zeros((dim, 2*dim))
    intercept = np.zeros((dim))
    if reg == "multitask" or alpha >= 0 or multivariate:
        if alpha >= 0:
            coeff, intercept = regressorLinguisticPrior(X, y, pos_pair, alpha, dim)
        elif reg == "multitask":
            lasso = regressor.MultiTaskLasso(alpha=5e-5)  #call multitask lasso directly here
            print "Fixing alpha to 5e-5"
            lasso.fit(X, y)
            print "Multitask R^2: %.3f"%(lasso.score(X, y))
            coeff = lasso.coef_
            intercept = lasso.intercept_
        else: #can only be multivariate
            coeff, intercept = regressorMultivariate(X, y, reg)
        for idx in range(0, dim): #re-assemble parameters in the right structure
            if concat:
                parameter[idx,:] = coeff[idx,:]
            else:
                parameter[idx,:,:] = coeff[idx,:].reshape((dim, dim)) if not diagonal else np.diag(coeff[idx,:])
    else: #for parallel/distributed estimation
        out_q = mp.Queue()
        procs = []
        chunksize = int(math.floor(dim / float(numProc)))
        for proc in range(numProc):
            end = dim if proc == numProc - 1 else (proc+1)*chunksize
            p = mp.Process(target=regressorParallel, args=(X, y, chunksize*proc, end, reg, out_q))
            procs.append(p)
            p.start()
        coefficients = []
        for proc in range(numProc): 
            coefficients += out_q.get()
        for p in procs:
            p.join()
        avgR2 = 0
        for coeff_idx_tuple in coefficients:
            idx, R2, coeff, inter = coeff_idx_tuple
            avgR2 += R2
            if concat:
                parameter[idx, :] = coeff
            else:
                parameter[idx, :, :] = coeff.reshape((dim, dim)) if not diagonal else np.diag(coeff)
            intercept[idx] = inter
        print "Parameter estimation complete and tensor has been formed"
        print "Average R2 across the %d regression problems: %.3f"%(dim, avgR2/dim)
    return parameter, intercept
        
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'cdfj:mnpP:r:')
    normalize = False
    diagonal = False
    filterDup = False
    concat = False
    ppdb = False
    reg = "lasso"
    jobs = 4
    alpha = -1
    multivariate = False
    for opt in opts:
        if opt[0] == '-n':
            normalize = True
        elif opt[0] == '-j':
            jobs = int(opt[1])
        elif opt[0] == '-d':
            diagonal = True
        elif opt[0] == '-f': 
            filterDup = True
        elif opt[0] == '-r':
            reg = opt[1]
            if not (reg == "lasso" or reg == "ridge" or reg == "lars" or reg == "elastic" or reg == "multitask"):
                sys.stderr.write("Error: regressor option not recognized; defaulting to 'lasso'\n")
                reg = "lasso"
        elif opt[0] == '-c': #concat model instead of outer product-based model
            concat = True
        elif opt[0] == '-p': #extract examples straight from PPDB
            ppdb = True
        elif opt[0] == '-P': #prior
            alpha = int(opt[1])
        elif opt[0] == '-m': #multivariate version of whatever algorithm chosen in -r
            multivariate = True
            if reg == "multitask":
                sys.stderr.write("Note: can only do multivariate regression on lasso, ridge, lars, or elastic")
                sys.exit()
    if diagonal and concat:
        sys.stderr.write("Error: cannot have diagonal parametrization and concatenative model together; setting diagonalization to false\n")
        diagonal = False
    if alpha >= 0 and (reg != "ridge" or not concat):
        sys.stderr.write("Error: linguistic regularization only works for L-2 prior (ridge regression) and concatenative models; setting regularizer to ridge and turning concatenation on\n")
        concat = True
        reg = "ridge"

    wordVecs = readVecFile(args[0], normalize)
    training_data = createTraining(args[1], wordVecs, filterDup) if ppdb else readTrainingFromFile(args[1], wordVecs, filterDup)
    print "Regressor chosen: %s"%reg
    parameters = {}
    for pos_pair in training_data:
        parameters[pos_pair] = learnParameters(training_data[pos_pair], pos_pair, jobs, diagonal, concat, reg, multivariate, alpha)
        print "Completed parameter learning for POS pair %s"%pos_pair
    cPickle.dump(parameters, open(args[2], "wb"))

if __name__ == "__main__":
    main()
