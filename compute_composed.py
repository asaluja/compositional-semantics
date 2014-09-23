#!/usr/bin/python -tt

'''
File: compute_composed.py
Date: July 25, 2014
Description: this script returns the composed representations for
a list of word pairs, given parameters.  Optionally, we can also
print out the pairwise similarities among the top N phrases, sorted by
frequency (i.e., order in which they are read in).  
Update (August 22, 2014): converted this script into a class representing
composed representations for better modularity/code reuse. 
This file can also be run by itself; it assumes the word vectors in the first
argument, the compositional model parameters as the second argument, and a 
list of POS-tagged phrases for which representations need to be computed in 
STDIN; if we are doing additive or multiplicative models, then we don't need
to POS-tag the phrases. 
'''

import sys, commands, string, cPickle, getopt, math
import pylab as plt
import matplotlib as mpl
import numpy as np

class CompoModel:
    def __init__(self, params_file, concat = False, normalize = True):
        self.wordVecs = {} #key is word, value is vector rep of word
        self.contextVecs = {}
        self.phraseVecs = {} #key is (phrase, pos_pair) tuple, value is vector rep of (phrase, pos_pair)
        self.parameters = cPickle.load(open(params_file, 'rb')) #dictionary; value is a tuple of parameters-intercept
        self.concat = concat
        self.normalize = normalize
        self.dimensions = self.parameters["X X"][0].shape[0]

    def readVecFile(self, filename, vecType = "word"):
        fh = open(filename, 'r')
        vecs = {}
        for line in fh:
            if len(line.strip().split()) > 2:
                word = line.strip().split()[0]
                rep = np.array([float(i) for i in line.strip().split()[1:]])
                vecs[word] = np.divide(rep, np.linalg.norm(rep)) if self.normalize else rep
                if self.normalize and np.linalg.norm(rep) == 0:
                    vecs[word] = np.zeros(len(rep))
        if vecType == "word":
            self.wordVecs = vecs
        else:
            self.contextVecs = vecs
            

    def checkVocab(self, phrase):
        words = phrase.split()
        if words[0] in self.wordVecs and words[1] in self.wordVecs:
            return True
        else:
            return False

    'This function assumes that checkVocab(phrase) returns true; that should be called before this'
    def computeComposedRep(self, phrase, pos_pair):    
        key = pos_pair if pos_pair in self.parameters else "X X"
        if (phrase, key) in self.phraseVecs:
            return self.phraseVecs[(phrase, key)]
        else:
            parameter, intercept = self.parameters[key]
            words = phrase.split()
            wordVec1 = self.wordVecs[words[0]]
            wordVec2 = self.wordVecs[words[1]]
            result = None
            if self.concat:
                argument = np.concatenate((wordVec1, wordVec2), axis=1)
                result = np.dot(parameter, argument.transpose())
            else:
                result = np.tensordot(wordVec2, parameter, axes=[0,2])
                result = np.dot(result, wordVec1)
            result += intercept
            if self.normalize:
                result = np.divide(result, np.linalg.norm(result))
            self.phraseVecs[(phrase, key)] = result
            return result

    def computeHeadedRep(self, phrase, pos_pair):
        phrase_words = phrase.split()
        pos_words = pos_pair.split()
        if "NN" in pos_words:
            if "JJ" in pos_words or sum([element == "NN" for element in pos_words]) == len(pos_words):
                return self.wordVecs[phrase_words[1]]
            elif "VV" in pos_words:
                return self.wordVecs[phrase_words[0]]
            else:
                return self.computeComposedRep(phrase, pos_pair)
        else:
            return self.computeComposedRep(phrase, pos_pair)

    'For point-wise additive/multiplicative models'
    def computeSimpleRep(self, phrase, operator):
        if (phrase, operator) in self.phraseVecs:
            return self.phraseVecs[(phrase, operator)]
        else:
            words = phrase.split()
            wordVec1 = self.wordVecs[words[0]]
            wordVec2 = self.wordVecs[words[1]]
            result = wordVec1 + wordVec2 if operator == "add" else np.multiply(wordVec1, wordVec2)
            self.phraseVecs[(phrase, operator)] = result
            return result

    def computePairwiseSimilarities(topN):
        for phrase, pos_pair in self.phraseVecs:
            phraseSims = []
            print "Phrase '%s' top %d similarities: "%(phrase, topN)
            for phrase2, pos_pair2 in self.phraseVecs.keys():
                if phrase != phrase2:
                    phraseRep1 = self.phraseVecs[(phrase, pos_pair)]
                    phraseRep2 = self.phraseVecs[(phrase2, pos_pair2)]
                    phraseSim = np.divide(np.dot(phraseRep1, phraseRep2), np.linalg.norm(phraseRep1) * np.linalg.norm(phraseRep2))
                    phraseSims.append((phrase2, phraseSim))
            phraseSims.sort(key = lambda x:x[1], reverse=True)
            topNPhraseSims = phraseSims[:topN]
            for phraseSim in topNPhraseSims:
                print "%s: %.3f\t"%(phraseSim[0], phraseSim[1]),
            print

    def printVector(self, phrase, rep):
        print "%s"%phrase,
        for idx in xrange(0, len(rep)):
            print " %.6f"%(rep[idx]),
        print

    def visualizeParameters(self, outFile_root, chartsPerRow, chartsPerCol):
        chartsPerCell = chartsPerRow * chartsPerCol
        numCharts = self.dimensions
        num_subplots = int(math.ceil(float(numCharts) / chartsPerCell))
        for pos_pair in self.parameters:
            pos_file = '_'.join(pos_pair.split())
            outFH = mpl.backends.backend_pdf.PdfPages(outFile_root + ".%s.pdf"%pos_file)
            parameter, intercept = self.parameters[pos_pair]
            for sp in xrange(num_subplots):
                chartNum = 0
                coordinate = sp*chartsPerCell
                f, axes_tuples = plt.subplots(chartsPerCol, chartsPerRow, sharey=True, sharex=True)
                while chartNum < chartsPerCell:
                    chartX = chartNum / chartsPerRow #truncates to nearest integer
                    chartY = chartNum % chartsPerRow
                    ax1 = axes_tuples[chartX][chartY]
                    cmap = plt.cm.get_cmap('RdBu')
                    maxVal = 0
                    minVal = 0
                    if coordinate < numCharts:                        
                        param = parameter[coordinate, :] if self.concat else parameter[coordinate, :, :]
                        maxVal = param[param!=0].max()
                        minVal = param[param!=0].min()
                        param[param==0] = np.nan
                        if self.concat:
                            param = np.reshape(param, (2, self.dimensions))
                        param = -param #negate values of parameters, since we want red to indicate high values
                        heatmap = np.ma.array(param, mask=np.isnan(param))
                        cmap.set_bad('w', 1.)
                        ax1.pcolor(heatmap, cmap=cmap, alpha=0.8)
                    else:
                        param = np.zeros((numCharts, numCharts))
                        ax1.pcolor(param, cmap=cmap, alpha=0.8)
                    ax1.set_title('Dim. %d; Max: %.3f; Min: %.3f'%(coordinate+1, maxVal, minVal))
                    ax1.set_xlim([0, numCharts])
                    if self.concat:
                        ax1.set_ylim([0, 2])
                        ax1.get_yaxis().set_ticks([])
                        ax1.set_ylabel('Words')
                        ax1.set_xlabel('Dimensions')
                    else:
                        ax1.set_ylim([0, numCharts])
                        ax1.set_ylabel('Left W')
                        ax1.set_xlabel('Right W')
                    chartNum += 1
                    coordinate += 1
                plt.tight_layout()
                outFH.savefig()
            outFH.close()

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'acmp:')
    topN = -1
    additive = False
    concat = False
    multiplicative = False
    for opt in opts:
        if opt[0] == '-p':
            topN = int(opt[1])
        elif opt[0] == '-a': 
            additive = True
        elif opt[0] == '-m':
            multiplicative = True
        elif opt[0] == '-c':
            concat = True
    if multiplicative and additive: #not possible
        sys.stderr.write("Error: Cannot have both '-a' and '-m' (additive and multiplicative) flags on!\n")
        sys.exit()
        
    model = CompoModel(args[1], concat, True)
    model.readVecFile(args[0])
    numExamples = 0
    numInVocab = 0
    numValidPOS = 0
    for line in sys.stdin:
        elements = line.strip().split()
        if len(elements) == 2:
            numExamples += 1
            words = [word_pos.split('_')[0] for word_pos in elements]
            pos_tags = [word_pos.split('_')[1] for word_pos in elements]
            phrase = ' '.join(words)
            if model.checkVocab(phrase): 
                numInVocab += 1
                rep = None
                if multiplicative:
                    rep = model.computeSimpleRep(phrase, "multiply")
                elif additive:
                    rep = model.computeSimpleRep(phrase, "add")
                else: #to do: change this section to handle generic POS pairs
                    contains_noun = "NN" in pos_tags or "NNS" in pos_tags or "NNP" in pos_tags or "NNPS" in pos_tags
                    if contains_noun:
                        if "JJ" in pos_tags or "JJR" in pos_tags or "JJS" in pos_tags:
                            if pos_tags[1] == "JJ" or pos_tags[1] == "JJR" or pos_tags[1] == "JJS": #wrong ordering, invert the ordering
                                words.reverse()
                                pos_tags.reverse()
                            rep = model.computeComposedRep(phrase, "JJ NN")
                            numValidPOS += 1
                        else:
                            valid = True
                            for pos_tag in pos_tags:
                                valid = valid & (pos_tag == "NN" or pos_tag == "NNS" or pos_tag == "NNPS")
                            if valid:
                                rep = model.computeComposedRep(phrase, "NN NN")
                                numValidPOS += 1
                if topN < 0:
                    printVector(phrase, rep)
    sys.stderr.write("Out of %d examples, %d are in the vocab, and %d of those have the correct POS sequence (if '-a' or '-m' flag on, then POS # doesn't matter)\n"%(numExamples, numInVocab, numValidPOS))
    if topN > 0: #i.e., pairwise similarities need to be computed
        model.computePairwiseSimilarities(topN)

if __name__ == "__main__":
    main()
