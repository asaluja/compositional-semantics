#!/usr/bin/python -t

'''
File:
Date:
Description:
'''

import sys, commands, string, cPickle, getopt
import numpy as np

def readVecFile(filename, normalize):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep)) if normalize else rep
    return repDict

def computeComposedRep(words, wordVecs, parameter, intercept):
    if words[0] in wordVecs and words[1] in wordVecs:
        wordVec1 = wordVecs[words[0]]
        wordVec2 = wordVecs[words[1]]
        result = np.tensordot(wordVec2, parameter, axes=[0,2])
        result = result.dot(wordVec1)
        result += intercept
        return result
    else:
        sys.stderr.write("Error! Word %s or %s is not in wordVecDictionary!\n"%(words[0], words[1]))

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
                    training_tuples.append((output, input_left, input_right))
                    #training_tuples.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right]))
                else: 
                    sys.stderr.write("Filtered example: %s ||| %s because of redundancy\n"%(output, elements[1]))
            else:
                training_tuples.append((output, input_left, input_right))
                #training_tuples.append((wordVecs[output], wordVecs[input_left], wordVecs[input_right], output, input_left, input_right))
        else:
            sys.stderr.write("Could not find one of the following words in the wordVec dictionary: %s, %s, or %s\n"%(output, input_left, input_right))
    print "Out of %d training examples, input-output triples exist for %d training examples"%(counter, len(training_tuples))
    return training_tuples

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'fln')
    normalize = False
    filterDup = False
    l2Dist = False
    for opt in opts:
        if opt[0] == '-n':
            normalize = True
        elif opt[0] == '-f':
            filterDup = True
        elif opt[0] == 'l': #l2 distance
            l2Dist = True
    wordVecs = readVecFile(args[0], normalize)
    parameter, intercept = cPickle.load(open(args[1], 'rb'))
    training_data = createTraining(wordVecs, filterDup)
    R2dist = {}
    for idx, triple in enumerate(training_data):
        word1 = triple[1]
        word2 = triple[2]
        outputWord = triple[0]
        #composedRep = computeComposedRep([word1, word2], wordVecs, parameter, intercept)
        #composedRep = np.divide(composedRep, np.linalg.norm(composedRep)) #composedRep should also be normalized
        phraseRep = wordVecs[outputWord]
        avgDist = 0.5*(np.linalg.norm(phraseRep - wordVecs[word1]) + np.linalg.norm(phraseRep - wordVecs[word2]))
        #sim = np.divide(np.dot(composedRep, phraseRep), np.linalg.norm(composedRep) * np.linalg.norm(phraseRep))
        #dist = np.linalg.norm(composedRep-phraseRep)
        R2dist[triple] = avgDist
    sorted_triples = sorted(R2dist, key=R2dist.get, reverse=True)
    for key in sorted_triples:
        print key, R2dist[key]
if __name__ == "__main__":
    main()
