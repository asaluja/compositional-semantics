#!/usr/bin/python -tt

'''
File: compute_composed.py
Date: July 25, 2014
Description: this script returns the composed representations for
a list of word pairs, given parameters.  Optionally, we can also
print out the pairwise similarities among the top N phrases, sorted by
frequency (i.e., order in which they are read in).  
'''

import sys, commands, string, cPickle, getopt
import numpy as np

def readVecFile(filename):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep))
    return repDict

#in this instance, when called, guaranteed for words to be in wordVecs
def computeComposedRep(words, wordVecs, parameter, intercept):    
    wordVec1 = wordVecs[words[0]]
    wordVec2 = wordVecs[words[1]]
    result = np.tensordot(wordVec2, parameter, axes=[0,2])
    result = result.dot(wordVec1)
    result += intercept
    return result

#simple multiplicative/additive model computation
def computeSimpleRep(additive, words, wordVecs):    
    wordVec1 = wordVecs[words[0]]
    wordVec2 = wordVecs[words[1]]
    result = None
    if additive:
        result = wordVec1 + wordVec2
    else:
        result = np.multiply(wordVec1, wordVec2)
    return result

def computePairwiseSimilarities(phraseVecs, topN):
    for phrase in phraseVecs:
        phraseSims = []
        print "Phrase '%s' top %d similarities: "%(phrase, topN)
        for phrase2 in phraseVecs.keys():
            if phrase != phrase2:
                phraseRep1 = phraseVecs[phrase]
                phraseRep2 = phraseVecs[phrase2]
                phraseSim = np.divide(np.dot(phraseRep1, phraseRep2), np.linalg.norm(phraseRep1) * np.linalg.norm(phraseRep2))
                phraseSims.append((phrase2, phraseSim))
        phraseSims.sort(key = lambda x:x[1], reverse=True)
        topNPhraseSims = phraseSims[:topN]
        for phraseSim in topNPhraseSims:
            print "%s: %.3f\t"%(phraseSim[0], phraseSim[1]),
        print

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'amp:')
    topN = -1
    additive = False
    multiplicative = False
    for opt in opts:
        if opt[0] == '-p':
            topN = int(opt[1])
        elif opt[0] == '-a': 
            additive = True
        elif opt[0] == '-m':
            multiplicative = True
    if multiplicative and additive: #not possible
        sys.stderr.write("Error: Cannot have both '-a' and '-m' (additive and multiplicative) flags on!\n")
        sys.exit()
    wordVecs = readVecFile(args[0])
    adjNounParam = cPickle.load(open(args[1], 'rb'))
    nounNounParam = cPickle.load(open(args[2], 'rb'))
    numExamples = 0
    numInVocab = 0
    numValidPOS = 0
    phraseVecs = {}
    for line in sys.stdin:
        elements = line.strip().split()
        if len(elements) == 2:
            numExamples += 1
            words = [word_pos.split('_')[0] for word_pos in elements]
            pos_tags = [word_pos.split('_')[1] for word_pos in elements]
            phrase = ' '.join(words)
            if words[0] in wordVecs and words[1] in wordVecs:
                numInVocab += 1
                rep = None
                if multiplicative:
                    rep = computeSimpleRep(False, words, wordVecs)
                    phraseVecs[phrase] = rep
                elif additive:
                    rep = computeSimpleRep(True, words, wordVecs)
                    phraseVecs[phrase] = rep
                else:
                    contains_noun = "NN" in pos_tags or "NNS" in pos_tags or "NNP" in pos_tags or "NNPS" in pos_tags
                    if contains_noun:
                        if "JJ" in pos_tags or "JJR" in pos_tags or "JJS" in pos_tags:
                            if pos_tags[1] == "JJ" or pos_tags[1] == "JJR" or pos_tags[1] == "JJS": #wrong ordering, invert the ordering
                                words.reverse()
                                pos_tags.reverse()
                            rep = computeComposedRep(words, wordVecs, adjNounParam[0], adjNounParam[1])
                            numValidPOS += 1
                            phraseVecs[phrase] = rep
                        else:
                            valid = True
                            for pos_tag in pos_tags:
                                valid = valid & (pos_tag == "NN" or pos_tag == "NNS" or pos_tag == "NNPS")
                            if valid:
                                rep = computeComposedRep(words, wordVecs, nounNounParam[0], nounNounParam[1])
                                numValidPOS += 1
                                phraseVecs[phrase] = rep
    sys.stderr.write("Out of %d examples, %d are in the vocab, and %d of those have the correct POS sequence (if '-a' or '-m' flag on, then POS # doesn't matter)\n"%(numExamples, numInVocab, numValidPOS))
    if topN > 0: #i.e., pairwise similarities need to be computed
        computePairwiseSimilarities(phraseVecs, topN)
    else:
        for phrase in phraseVecs:
            print "%s"%(phrase),
            for val in np.nditer(phraseVecs[phrase]):
                print " %.6f"%val,
            print 

if __name__ == "__main__":
    main()
