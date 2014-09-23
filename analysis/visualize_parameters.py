#!/usr/bin/python -tt

'''
File: visualize_parameters.py
Date: July 21, 2014
Description: this script visualizes the non-zero values of the
combiner function parameters.  
Usage: python visualize_parameters.py parameters outFile numChartsInRow numChartsInCol
Update (September 23, 2014): modified the code to handle groups of parameters at a time;
each combiner function is written out to a separate file.  Optional flag for concatenative
model also added. 
'''

import sys, commands, string, getopt, os
lib_path = os.path.abspath('/usr0/home/avneesh/compositional-models/code/')
sys.path.append(lib_path)
from compute_composed import *

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'c')
    concat = False
    for opt in opts:
        if opt[0] == '-c':
            concat = True
    model = CompoModel(args[0], concat, True)
    outFile_root = args[1]
    chartsPerRow = int(args[2])
    chartsPerCol = int(args[3])
    model.visualizeParameters(outFile_root, chartsPerRow, chartsPerCol)

if __name__ == "__main__":
    main()
