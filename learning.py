#!/usr/bin/python -tt

'''
File: learning.py
Date: July 2, 2014
Description: this script takes as a list of word pairs, as well as the
vector representations of those words, and sets up a regression problem
to learn the parameters in our composition function. 
Usage: python learning.py wordVectorsIn ParametersOut < training_data
'''

import sys, commands, string, getopt, cPickle, math
import numpy as np
import multiprocessing as mp
import sklearn.linear_model as regressor

def readVecFile(filename, normalize):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep)) if normalize else rep
    return repDict

def createTraining(wordVecs, filterDup):
    training_tuples = []
    counter = 0
    for line in sys.stdin:
        counter += 1
        elements = line.strip().split(' ||| ')
        output = elements[0]
        input_left = elements[1].split()[0]
        input_right = elements[1].split()[1]
        if output in wordVecs and input_left in wordVecs and input_right in wordVecs: #may want to print out some information on the 'else' condition
            if filterDup:
                if input_left != output and input_right != output:
                    training_tuples.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right]))
                else: 
                    sys.stderr.write("Filtered example: %s ||| %s because of redundancy\n"%(output, elements[1]))
            else:
                training_tuples.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right]))
        else:
            sys.stderr.write("Could not find one of the following words in the wordVec dictionary: %s, %s, or %s\n"%(output, input_left, input_right))
    print "Out of %d training examples, input-output triples exist for %d training examples"%(counter, len(training_tuples))
    return training_tuples

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
        reg.fit(data, labels[:,idx])        
        print "Dimension %d Alpha selected: %.3g"%(idx, reg.alpha_) #use this for CV experiments
        R2 = reg.score(data, labels[:,idx])
        print "Dimension %d R^2 on data: %.3f"%(idx, R2)
        print "Dimension %d Number of non-zero values in coefficients: %d"%(idx, (reg.coef_ != 0).sum())
        coefficients.append((idx, R2, reg.coef_, reg.intercept_))
    out_q.put(coefficients)

def learnParameters(training_data, numProc, diagonal, reg):
    numSamples = len(training_data)
    dim = len(training_data[0][0])
    P = dim if diagonal else dim * dim
    print "Number of training examples: %d; Number of regression problems: %d; Number of covariates: %d"%(numSamples, dim, P)
    y = np.zeros((numSamples, dim))
    X = np.zeros((numSamples, P))
    for idx, triple in enumerate(training_data): 
        y[idx,:] = triple[0].transpose()
        X[idx,:] = np.hstack(np.outer(triple[1], triple[2])) if not diagonal else np.diagonal(np.outer(triple[1], triple[2]))
    print "Completed assembling training data into regression format.  Now starting regression."
    '''
    lasso = regressor.MultiTaskLasso(alpha=5e-05)
    lasso.fit(X, y)
    print "R^2: %.3f"%(lasso.score(X, y))
    coeff = lasso.coef_
    intercept = lasso.intercept_
    for idx in range(0, dim):
        parameter[idx,:,:] = coeff[idx,:].reshape((dim, dim))
    '''
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
    parameter = np.zeros((dim, dim, dim))
    intercept = np.zeros((dim))
    avgR2 = 0
    for coeff_idx_tuple in coefficients:
        idx, R2, coeff, inter = coeff_idx_tuple
        avgR2 += R2
        #handle diagonal here
        parameter[idx, :, :] = coeff.reshape((dim, dim)) if not diagonal else np.diag(coeff)
        intercept[idx] = inter
    print "Parameter estimation complete and tensor has been formed"
    print "Average R2 across the %d regression problems: %.3f"%(dim, avgR2/dim)
    return parameter, intercept
        
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'dfj:nr:')
    normalize = False
    diagonal = False
    filterDup = False
    reg = "lasso"
    jobs = 4
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
            if not (reg == "lasso" or reg == "ridge" or reg == "lars" or reg == "elastic"):
                sys.stderr.write("Error: regressor option not recognized; defaulting to 'lasso'\n")
                reg = "lasso"
    wordVecs = readVecFile(args[0], normalize)
    training_data = createTraining(wordVecs, filterDup)
    print "Regressor chosen: %s"%reg
    parameter_intercept_tuple = learnParameters(training_data, jobs, diagonal, reg)
    cPickle.dump(parameter_intercept_tuple, open(args[1], "wb"))

if __name__ == "__main__":
    main()
