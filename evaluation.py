#!/usr/bin/python -tt

'''
File: evaluation.py
Date: July 3, 2014 
Description: this file takes as an input the parameters estimated in learning.py,
and also a list of word pairs which can be different syntactic combinations 
(JJ-NN, VBZ-NN, etc.).  The particular combination is defined by the flag passed.  
The specific word pairs are created by post-processing the Mitchell & Lapata 2010
dataset, 

'''

import sys, commands, string, cPickle, getopt
import numpy as np
import scipy.stats as stats

def readVecFile(filename, normalize):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep)) if normalize else rep
    return repDict

def readHumanSimilarities(selector):
    humanDict = {}
    for line in sys.stdin:
        elements = line.strip().split()
        if elements[1] == selector:
            phrase1 = ' '.join(elements[3:5])
            phrase2 = ' '.join(elements[5:7])
            sim = int(elements[-1])
            sims = humanDict[elements[0]] if elements[0] in humanDict else []
            sims.append((phrase1, phrase2, sim))
            humanDict[elements[0]] = sims
    return humanDict

def computeComposedRep(phrase, wordVecs, parameter, intercept, mult):
    words = phrase.split()
    if words[0] in wordVecs and words[1] in wordVecs:
        wordVec1 = wordVecs[words[0]]
        wordVec2 = wordVecs[words[1]]
        result = None
        if mult:
            result = np.multiply(wordVec1, wordVec2)
        else:                              
            result = np.tensordot(wordVec2, parameter, axes=[0,2])
            result = result.dot(wordVec1)
            result += intercept
        return result
    else:
        sys.stderr.write("Error! Word %s or %s is not in wordVecDictionary!\n"%(words[0], words[1]))

def computeCorrelation(humanSims, wordVecs, parameter, intercept, mult, extreme):
    composedReps = {}
    rhoVec = []
    numRatings = []
    for subject in humanSims:
        human_ratings = []
        model_ratings = []
        for phrase1, phrase2, sim in humanSims[subject]:
            phraseRep1 = composedReps[phrase1] if phrase1 in composedReps else computeComposedRep(phrase1, wordVecs, parameter, intercept, mult)
            composedReps[phrase1] = phraseRep1
            phraseRep2 = composedReps[phrase2] if phrase2 in composedReps else computeComposedRep(phrase2, wordVecs, parameter, intercept, mult)
            composedReps[phrase2] = phraseRep2
            phraseSim = np.divide(np.dot(phraseRep1, phraseRep2), np.linalg.norm(phraseRep1) * np.linalg.norm(phraseRep2))
            if extreme:
                if sim < 3 or sim > 5:
                    human_ratings.append(sim)
                    model_ratings.append(phraseSim)
            else:
                human_ratings.append(sim)
                model_ratings.append(phraseSim)
        rho = stats.spearmanr(human_ratings, model_ratings)
        rhoVec.append(rho[0])
        numRatings.append(len(human_ratings))
    print "Average rho across %d subjects: %.3f"%(len(humanSims), sum(rhoVec) / len(humanSims))
    print "Average number of ratings across %d subjects: %.3f"%(len(humanSims), float(sum(numRatings)) / len(numRatings))
            

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'ejmnN')
    normalize = False
    selector = ""
    mult = False
    extreme = False
    for opt in opts:
        if opt[0] == '-n':
            normalize = True        
        elif opt[0] == '-j':
            selector = "adjectivenouns"
        elif opt[0] == '-m':
            mult = True
        elif opt[0] == '-e':
            extreme = True
        elif opt[0] == '-N':
            selector = "compoundnouns"
    wordVecs = readVecFile(args[0], normalize)
    parameter, intercept = cPickle.load(open(args[1], 'rb'))
    humanSims = readHumanSimilarities(selector)
    computeCorrelation(humanSims, wordVecs, parameter, intercept, mult, extreme)
    

if __name__ == "__main__":
    main()
