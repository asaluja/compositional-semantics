#!/usr/bin/python -tt

'''
File: quantize_annotations.py
Date: November 20, 2014
Description: script that takes human annotations (numerical scores) of whatever
in STDIN, and the number of bins as an argument, and outputs the quantized
version of the data. 
'''

import sys, commands, string
import numpy as np
from scipy import stats

def main():
    phrases = []
    scores = []
    for line in sys.stdin:
        elements = line.strip().split('\t')
        if elements[0] == 'EN_ADJ_NN':
            phrases.append(elements[1])
            scores.append(int(elements[2]))
    numBins = int(sys.argv[1])
    numElementsPerBin = float(len(scores)) / numBins
    probs = np.array(range(numBins+1))*numElementsPerBin
    probs /= float(len(scores))
    #bins = sp.stats.mstats.mquantiles(scores, probs)
    bins = stats.mstats.mquantiles(scores, probs)
    binned_scores = np.digitize(scores, bins)
    for idx, phrase in enumerate(phrases):
        print "EN_ADJ_NN\t%s\t%d"%(phrase, binned_scores[idx])
                         
    

if __name__ == "__main__":
    main()
