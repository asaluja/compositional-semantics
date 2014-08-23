#!/usr/bin/python -tt

'''
File: non_comp_detect.py
Date: August 21, 2014
Description: 
'''

import sys, commands, string, getopt, math, gzip
import numpy as np
import extract_training
from compute_composed import *

class Noise:
    def __init__(self, filename, negSamples):
        self.words = []
        counts = []
        fh = open(filename, 'rb')
        for line in fh:
            elements = line.strip().split('\t')
            self.words.append(elements[0])
            counts.append(float(elements[1]))
        normalizer = sum(counts)
        self.probs = [count/normalizer for count in counts]
        self.negSamples = negSamples
    
    def sampleWords(self):
        return np.random.choice(self.words, self.negSamples, replace=False, p=self.probs)

    def filterStopWords(self, filename, numStop):
        fh = open(filename, 'rb')
        counter = 0
        self.stopwords = []
        for line in fh:
            elements = line.strip().split('\t')
            self.stopwords.append(elements[0])
            counter += 1
            if counter == numStop:
                break
        for word in self.stopwords:
            idx = self.words.index(word)
            self.words.pop(idx)
            self.probs.pop(idx)
        normalizer = sum(self.probs)
        self.probs = [prob/normalizer for prob in self.probs]

    def checkStopWord(self, word):
        return word in self.stopwords
            
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def scoreLogReg(context, phrase, phraseRep, model, noise):
    victories = []
    for word in context:
        if word in model.wordVecs: 
            wordRep = model.wordVecs[word]
            score = sigmoid(np.dot(wordRep, phraseRep))
            print "phrase: %s; context: %s; score: %.3f"%(phrase, word, score)
            noiseWords = noise.sampleWords()
            count = 0
            for noiseWord in noiseWords:
                if noiseWord in model.wordVecs:
                    noiseWordRep = model.wordVecs[noiseWord]
                    noiseScore = sigmoid(np.dot(noiseWordRep, phraseRep))
                    print "noise word: %s; score: %.3f"%(noiseWord, noiseScore)
                    if score > noiseScore:
                        count += 1
            if count > len(noiseWords) / 2:
                victories.append(True)
    if False not in victories:
        print "Success!"

def scoreCosineSim(phrase, phraseRep, model):
    words = phrase.split()
    wordVec1 = model.wordVecs[words[0]]
    wordVec2 = model.wordVecs[words[1]]
    cosSim1 = np.divide(np.dot(wordVec1, phraseRep), np.linalg.norm(wordVec1) * np.linalg.norm(phraseRep))
    cosSim2 = np.divide(np.dot(wordVec2, phraseRep), np.linalg.norm(wordVec2) * np.linalg.norm(phraseRep))
    return 0.5*(cosSim1 + cosSim2)

def checkArity(rule):
    return len(re.findall(r'\[([^]]+)\]', rule))

def processRules(grammar_fh):
    seen_rules = []
    preterm_rules = []
    for rule in grammar_fh: 
        src_rule = rule.strip().split(' ||| ')[1]
        if not src_rule in seen_rules:
            seen_rules.append(src_rule)
            if checkArity(src_rule) == 0:
                preterm_rules.append(src_rule)
    return preterm_rules

def extractContext(words, start_idx, end_idx, model, noise):
    start = start_idx-1
    left_context = ""
    while start > -1:
        left_context = words[start]
        if not noise.checkStopWord(left_context) and left_context in model.wordVecs:
            break
        start -= 1
    if start == -1: #can try replacing this with 'else'
        left_context = "<s>"
    end = end_idx
    while end < len(words):
        right_context = words[end]
        if not noise.checkStopWord(right_context) and right_context in model.wordVecs:
            break
        end += 1
    if end == len(words): #can try replacing this with 'else'
        right_context = "</s>"
    return (left_context, right_context)

def containsSequence(subseq, seq):
    for i in xrange(len(seq)-len(subseq)+1):
        for j in xrange(len(subseq)):
            if seq[i+j] != subseq[j]:
                break
        else:
            return i, i+len(subseq)
    return False

def main():    
    (opts, args) = getopt.getopt(sys.argv[1:], 'cs:')
    concat = False
    negSamples = 10
    for opt in opts:
        if opt[0] == '-c':
            concat = True
        elif opt[0] == '-s':
            negSamples = int(opt[1])
    model = CompoModel(args[1], concat, True)
    model.readVecFile(args[0])
    noise = Noise(args[2], negSamples)
    noise.filterStopWords(args[3], 20)
    grammar_loc = args[4]
    line_counter = 0
    for line in sys.stdin: #each line is a POS-tagged sentence (sequence of word#POS pairs)
        words, pos_tags = zip(*[word_pos.split('#') for word_pos in line.strip().split()])
        phrases = processRules(gzip.open(grammar_loc + "grammar.%d.gz"%line_counter, 'rb'))
        line_counter += 1
        for phrase in phrases:
            phrase_words = phrase.split()
            start, end = containsSequence(phrase_words, words)
            phrase_pos = [extract_training.collapsePOS(pos) for pos in pos_tags[start:end+1]]
            #then need to figure out best way to combine them
            print "Phrase %s; POS %s"%(phrase, ' '.join(phrase_pos))
            '''
            midPoint = 7
            bigram = ' '.join(words[midPoint:midPoint+2])
            bigram_pos = ' '.join(pos_tags[midPoint:midPoint+2])
            print "Bigram %s; POS %s"%(bigram, bigram_pos)
            '''
            phraseRep = model.computeComposedRep(bigram, bigram_pos)
            context = extractContext(words, midPoint, midPoint+2, model, noise)
            print context
            scoreLogReg(context, bigram, phraseRep, model, noise) 
     
if __name__ == "__main__":
    main()
