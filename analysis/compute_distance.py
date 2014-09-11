#!/usr/bin/python -tt

'''
File: compute_distance.py
Date: September 11, 2014
Description: given two representations (in two separate files), 
this script uses the keys in the first file as the set of keys
to search for in the second file.  It then computes the distances
between the vectors, sorts, and prints these distances out. 
'''

import sys, commands, string
import numpy as np

def readVecFile(filename, normalize):
    fh = open(filename, 'r')
    repDict = {}
    for line in fh:
        word = line.strip().split()[0]
        rep = np.array([float(i) for i in line.strip().split()[1:]])
        repDict[word] = np.divide(rep, np.linalg.norm(rep)) if normalize else rep
    return repDict

def main():
    ref_file = open(sys.argv[1], 'rb')
    ref_vecs = {}
    for line in ref_file:
        elements = line.strip().split()
        phrase = elements[0]
        vector = [float(a) for a in elements[1:]]
        ref_vecs[phrase] = vector
    other_vecs = readVecFile(sys.argv[2], True)
    for phrase in ref_vecs:
        phraseToPrint = ' '.join(phrase.split('_'))
        if phrase in other_vecs:
            distance = np.linalg.norm(ref_vecs[phrase] - other_vecs[phrase])
            print "%s:%.6f"%(phraseToPrint, distance)
        else:
            sys.stderr.write("%s not in training\n"%phraseToPrint)

if __name__ == "__main__":
    main()
