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

class PhraseTree:        
    def __init__(self, wordVector, tag, left=None, right=None):
        self.vector = wordVector
        self.tag = tag
        self.left = left
        self.right = right
        self.parent = None
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

class CompoModel:
    def __init__(self, params_file, concat = False, normalize = True, headed = False, rightBranch = False):
        self.wordVecs = {} #key is word, value is vector rep of word
        self.contextVecs = {}
        self.phraseVecs = {} #key is (phrase, pos_pair) tuple, value is vector rep of (phrase, pos_pair)
        self.parameters = cPickle.load(open(params_file, 'rb')) #dictionary; value is a tuple of parameters-intercept
        self.concat = concat
        self.headed = headed
        self.normalize = normalize
        self.dimensions = self.parameters["X X"][0].shape[0]
        self.rightBranch = rightBranch

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
        for word in words:
            if word not in self.wordVecs:
                return False
        else:
            return True

    def checkTag(self, prev_tag, tag):
        new_tag = "NN" if tag == "NN" and (prev_tag == "JJ" or prev_tag == "DT" or prev_tag == "NN") else "X"
        return new_tag

    def constructPhraseTree(self, phrase, pos_seq):
        maxChildren = 2
        numChildren = 0
        root = None
        prevNode = None
        words = phrase.split()
        pos_tags = pos_seq.split()
        if self.rightBranch:
            words.reverse()
            pos_tags.reverse()
        for idx, word in enumerate(words):
            tag = pos_tags[idx]
            node = PhraseTree(self.wordVecs[word], tag)
            numChildren += 1
            if numChildren == maxChildren: #create parent
                parent_tag = self.checkTag(prevNode.tag, tag)
                parent = PhraseTree(np.zeros(self.dimensions), parent_tag, prevNode, node) if not self.rightBranch else PhraseTree(np.zeros(self.dimensions), parent_tag, node, prevNode)
                if root is None: #previously unassigned
                    root = parent
                else:
                    new_root_tag = self.checkTag(root.tag, tag)
                    new_root = PhraseTree(np.zeros(self.dimensions), new_root_tag, root, parent) if not self.rightBranch else PhraseTree(np.zeros(self.dimensions), new_root_tag, parent, root)
                    root = new_root
                numChildren = 0
            prevNode = node
        if prevNode.parent is None: #now, need to attach any unattached nodes
            if root is None:
                return prevNode
            else:
                new_root_tag = self.checkTag(root.tag, prevNode.tag) #root is None condition for unattached single nodes (unigram phrases)
                new_root = PhraseTree(np.zeros(self.dimensions), new_root_tag, root, prevNode) if not self.rightBranch else PhraseTree(np.zeros(self.dimensions), new_root_tag, prevNode, root)
                root = new_root            
        return root

    def computePhraseTreeRep(self, phraseTree):
        if phraseTree is None: return
        if phraseTree.left is None and phraseTree.right is None: #leaf
            return phraseTree.vector
        self.computePhraseTreeRep(phraseTree.left)
        self.computePhraseTreeRep(phraseTree.right)        
        wordVec1 = phraseTree.left.vector
        wordVec2 = phraseTree.right.vector
        pos_tags = [phraseTree.left.tag, phraseTree.right.tag]        
        if self.headed:
            headIdx = self.computeHeadedRep(pos_tags)
            if headIdx > -1: #if headIdx == -1, then we computeComposed anyway
                phraseTree.vector = wordVec1 if headIdx == 0 else wordVec2
                return phraseTree.vector
        key = ' '.join(pos_tags)
        key = "X X" if key not in self.parameters else key
        parameter, intercept = self.parameters[key]                        
        if self.concat:
            argument = np.concatenate((wordVec1, wordVec2), axis=1)                       
            result = np.dot(parameter, argument.transpose())                       
        else:
            result = np.tensordot(wordVec2, parameter, axes=[0,2])
            result = np.dot(result, wordVec1)
        result += intercept                       
        if self.normalize:
            result = np.divide(result, np.linalg.norm(result))
        phraseTree.vector = result                       
        return result

    def computeHeadedRep(self, pos_words):
        if "NN" in pos_words:                           
            if "JJ" in pos_words or "DT" in pos_words or sum([element == "NN" for element in pos_words]) == len(pos_words): 
                return 1
            elif "VV" in pos_words:
                return 0
            else:
                return -1
        else:
            return -1            

    'This function assumes that checkVocab(phrase) returns true; that should be called before this'
    def computeComposedRep(self, phrase, pos_seq):    
        if (phrase, pos_seq) in self.phraseVecs:
            return self.phraseVecs[(phrase, pos_seq)]
        else:
            phraseTree = self.constructPhraseTree(phrase, pos_seq)            
            result = self.computePhraseTreeRep(phraseTree)
            self.phraseVecs[(phrase, pos_seq)] = result
            return result


    'For point-wise additive/multiplicative models'
    def computeSimpleRep(self, phrase, operator):
        if (phrase, operator) in self.phraseVecs:
            return self.phraseVecs[(phrase, operator)]
        else:
            words = phrase.split()
            result_dim = self.wordVecs[words[0]].shape
            result = np.zeros(result_dim) if operator == "add" else np.ones(result_dim)
            for word in words:
                result = result + self.wordVecs[word] if operator == "add" else np.multiply(result, self.wordVecs[word])
            if self.normalize:
                result = np.divide(result, np.linalg.norm(result))
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
            if self.concat: 
                left_mat = -parameter[:,:self.dimensions]
                left_mat[left_mat==0] = np.nan                
                right_mat = -parameter[:,self.dimensions:]
                right_mat[right_mat==0] = np.nan                
                f, axes_tuples = plt.subplots(1, 1)
                ax1 = axes_tuples
                cmap = plt.cm.get_cmap('RdBu')
                heatmap = np.ma.array(left_mat, mask=np.isnan(left_mat))
                cmap.set_bad('w', 1.)
                ax1.pcolor(heatmap, cmap=cmap, alpha=0.8)
                ax1.set_title('First Word')
                plt.tight_layout()
                outFH.savefig()
                f, axes_tuples = plt.subplots(1,1)
                ax1 = axes_tuples
                cmap = plt.cm.get_cmap('RdBu')
                heatmap = np.ma.array(right_mat, mask=np.isnan(right_mat))
                cmap.set_bad('w', 1.)
                ax1.pcolor(heatmap, cmap=cmap, alpha=0.8)
                ax1.set_title('Second Word')
                plt.tight_layout()
                outFH.savefig()
                outFH.close()
            else:
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
                            param = parameter[coordinate, :, :]
                            maxVal = param[param!=0].max()
                            minVal = param[param!=0].min()
                            param[param==0] = np.nan
                            param = -param #negate values of parameters, since we want red to indicate high values
                            heatmap = np.ma.array(param, mask=np.isnan(param))
                            cmap.set_bad('w', 1.)
                            ax1.pcolor(heatmap, cmap=cmap, alpha=0.8)
                        else:
                            param = np.zeros((numCharts, numCharts))
                            ax1.pcolor(param, cmap=cmap, alpha=0.8)
                        ax1.set_title('Dim. %d; Max: %.3f; Min: %.3f'%(coordinate+1, maxVal, minVal))
                        ax1.set_xlim([0, numCharts])
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
