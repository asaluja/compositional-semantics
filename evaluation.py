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
        if selector == "adjectivenouns" or selector == "compoundnouns":
            elements = line.strip().split()        
            if elements[1] == selector:
                phrase1 = ' '.join(elements[3:5])
                phrase2 = ' '.join(elements[5:7])
                sim = int(elements[-1])
                sims = humanDict[elements[0]] if elements[0] in humanDict else []
                sims.append((phrase1, phrase2, sim))
                humanDict[elements[0]] = sims
        elif selector == "NonCompAdjNoun":
            elements = line.strip().split('\t')
            if elements[0] == "EN_ADJ_NN":
                humanDict[elements[1]] = float(elements[2])
        elif selector == "NonCompNounNoun":
            elements = line.strip().split('\t')
            phrase = ' '.join(elements[0].split('_'))
            scores = humanDict[elements[1]] if elements[1] in humanDict else []
            if elements[3] == "accepted":
                scores.append((phrase, int(elements[4]))) #for each participant, store phrase and score
                humanDict[elements[1]] = scores #humanDict is indexed by participantID
    return humanDict

def readComputedScores(filename):
    computed_scores = {}
    fh = open(filename, 'rb')
    for line in fh:
        phrase, score = line.strip().split(' ||| ')
        computed_scores[phrase] = float(score)
    return computed_scores

def computeSimilarityCorrelation(humanSims, model, mult, extreme, divergent, selector):
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

def computeNonCompCorrelation(humanSims, scores, selector):    
    if selector == "NonCompAdjNoun":
        human_scores = []
        learned_scores = []
        for phrase in scores:
            if phrase in humanSims:
                learned_scores.append(scores[phrase])
                human_scores.append(humanSims[phrase])
        rho = stats.spearmanr(human_scores, learned_scores)
        print "Spearman rho: %.3f; over %d examples"%(rho[0], len(human_scores))
    else: #for noun-noun, it's a bit more complicated
        rhoVec = []
        for subject in humanSims:
            human_scores = []
            learned_scores = []
            scores_only = [phrase_score[1] for phrase_score in humanSims[subject]]
            allSame = len(set(scores_only)) <= 1
            if not allSame:
                for phrase, score in humanSims[subject]: 
                    if phrase in scores:
                        human_scores.append(score)
                        learned_scores.append(scores[phrase])
                rho = stats.spearmanr(human_scores, learned_scores)
                rhoVec.append(rho[0])
        print "Average rho across %d subjects: %.3f"%(len(rhoVec), sum(rhoVec) / len(rhoVec))

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'cdejJmnN')
    selector = ""
    mult = False
    extreme = False
    concat = False
    divergent = False
    for opt in opts:
        if opt[0] == '-n':
            selector = "compoundnouns"
        elif opt[0] == '-j':
            selector = "adjectivenouns"
        elif opt[0] == '-J':
            selector = "NonCompAdjNoun"
        elif opt[0] == '-N':
            selector = "NonCompNounNoun"
        elif opt[0] == '-m':
            mult = True
        elif opt[0] == '-c':
            concat = True
        elif opt[0] == '-d':
            divergent = True
        elif opt[0] == '-e':
            extreme = True
    computeSimCorr = (selector == "compoundnouns") or (selector == "adjectivenouns")
    humanSims = readHumanSimilarities(selector)
    if computeSimCorr:
        model = CompoModel(args[1], concat, True)
        model.readVecFile(args[0])
        computeSimilarityCorrelation(humanSims, model, mult, extreme, divergent, selector)
    else:
        scores = readComputedScores(args[1])
        computeNonCompCorrelation(humanSims, scores, selector) 

if __name__ == "__main__":
    main()
