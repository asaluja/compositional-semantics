#!/usr/bin/python -tt

'''
File: evaluation.py
Date: July 3, 2014 
Description: this file takes as an input the parameters estimated in learning.py,
and also a list of word pairs which can be different syntactic combinations 
(JJ-NN, VBZ-NN, etc.).  The particular combination is defined by the flag passed.  
The specific word pairs are created by post-processing the Mitchell & Lapata 2010
dataset.
Update (August 22, 2014): modified to interface with CompoModel class
'''

import sys, commands, string, getopt
import numpy as np
import scipy.stats as stats
from compute_composed import *

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

def computeCorrelation(humanSims, model, mult, extreme, divergent, selector):
    pos_pair = "JJ NN" if selector == "adjectivenouns" else "NN NN"
    rhoVec = []
    numRatings = []
    for subject in humanSims:
        human_ratings = []
        model_ratings = []
        phrase_pairs = []
        for phrase1, phrase2, sim in humanSims[subject]:
            phrase_pairs.append((phrase1, phrase2))
            if model.checkVocab(phrase1) and model.checkVocab(phrase2): 
                phraseRep1 = model.computeSimpleRep(phrase1, "multiply") if mult else model.computeComposedRep(phrase1, pos_pair)
                phraseRep2 = model.computeSimpleRep(phrase2, "multiply") if mult else model.computeComposedRep(phrase2, pos_pair)                
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
        if divergent:
            human_rankings = stats.rankdata(human_ratings)
            model_rankings = stats.rankdata(model_ratings)
            print "Subject: %s"%subject
            for idx, rank in enumerate(human_rankings):
                printPair = False
                if np.absolute(rank-model_rankings[idx]) < len(human_rankings) / 18: 
                    print "Small difference between human and model!"
                    printPair = True
                elif np.absolute(rank-model_rankings[idx]) > len(human_rankings) / 2:
                    print "Big difference between human and model!"
                    printPair = True
                if printPair:
                    print "Phrase pair: '%s' and '%s'"%(phrase_pairs[idx][0], phrase_pairs[idx][1])
                    print "Human rating and rank: %d/%d"%(human_ratings[idx], rank)
                    print "Model rating and rank: %.3f/%d"%(model_ratings[idx], model_rankings[idx])
    print "Average rho across %d subjects: %.3f"%(len(humanSims), sum(rhoVec) / len(humanSims))
    print "Average number of ratings across %d subjects: %.3f"%(len(humanSims), float(sum(numRatings)) / len(numRatings))

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'cdejmnN')
    normalize = False
    selector = ""
    mult = False
    extreme = False
    concat = False
    divergent = False
    for opt in opts:
        if opt[0] == '-n':
            normalize = True        
        elif opt[0] == '-j':
            selector = "adjectivenouns"
        elif opt[0] == '-N':
            selector = "compoundnouns"
        elif opt[0] == '-m':
            mult = True
        elif opt[0] == '-c':
            concat = True
        elif opt[0] == '-d':
            divergent = True
        elif opt[0] == '-e':
            extreme = True
    model = CompoModel(args[1], concat, normalize)
    model.readVecFile(args[0])
    humanSims = readHumanSimilarities(selector)
    computeCorrelation(humanSims, model, mult, extreme, divergent, selector)

if __name__ == "__main__":
    main()
