#!/usr/bin/python -tt

'''
File: non_comp_detect.py
Date: August 21, 2014
Description: 
'''

import sys, commands, string, getopt, math, gzip, re
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
            #print "phrase: %s; context: %s; score: %.3f"%(phrase, word, score)
            noiseWords = noise.sampleWords()
            count = 0
            for noiseWord in noiseWords:
                if noiseWord in model.wordVecs:
                    noiseWordRep = model.wordVecs[noiseWord]
                    noiseScore = sigmoid(np.dot(noiseWordRep, phraseRep))
                    #print "noise word: %s; score: %.3f"%(noiseWord, noiseScore)
                    if score > noiseScore:
                        count += 1
            if count > len(noiseWords) / 2:
                victories.append(True)
            else:
                victories.append(False)
    if sum(victories) >= len(victories) / 2:
        return True
    else:
        return False

def scoreSkipGram(context, phrase, phraseRep, model, noise):
    aggregate_score = 0.0
    numContext = 0
    for word in context:
        if word in model.contextVecs:
            numContext += 1
            wordRep = model.contextVecs[word]
            aggregate_score += np.dot(wordRep, phraseRep)
            #aggregate_score += math.exp(np.dot(wordRep, phraseRep)) #unnormalized
    avg_score = aggregate_score / numContext
    print "Phrase '%s':%.3f"%(phrase, avg_score)
    #print "Context is: "
    #print context

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
            if checkArity(src_rule) == 0 and len(src_rule.split()) == 2: #second condition is temporary
                preterm_rules.append(src_rule)
    return preterm_rules

def extractContext(words, start_idx, end_idx, context_size, model, noise):
    start = start_idx-1
    left_context = []
    while start > -1:
        left_word = words[start]
        if left_word in model.contextVecs:
        #if not noise.checkStopWord(left_word) and left_word in model.contextVecs:
            left_context.append(left_word)
        if len(left_context) == context_size:
            break
        start -= 1
    end = end_idx
    right_context = []
    while end < len(words):
        right_word = words[end]
        if right_word in model.contextVecs:
        #if not noise.checkStopWord(right_word) and right_word in model.contextVecs:
            right_context.append(right_word)
        if len(right_context) == context_size:
            break
        end += 1
    return left_context + right_context

def containsSequence(subseq, seq):
    for i in xrange(len(seq)-len(subseq)+1):
        for j in xrange(len(subseq)):
            if seq[i+j] != subseq[j]:
                break
        else:
            return i, i+len(subseq)
    return -1, -1

def main():    
    (opts, args) = getopt.getopt(sys.argv[1:], 'cC:s:')
    concat = False
    negSamples = 10
    numContext = 1
    for opt in opts:
        if opt[0] == '-c':
            concat = True
        elif opt[0] == '-s':
            negSamples = int(opt[1])
        elif opt[0] == '-C':
            numContext = int(opt[1])
    model = CompoModel(args[2], concat, True)
    model.readVecFile(args[0], "word")
    model.readVecFile(args[1], "context")
    noise = Noise(args[3], negSamples)
    noise.filterStopWords(args[4], 20)
    grammar_loc = args[5]
    line_counter = 0
    for line in sys.stdin: #each line is a POS-tagged sentence (sequence of word#POS pairs)
        words, pos_tags = zip(*[word_pos.split('#') for word_pos in line.strip().split()])
        phrases = processRules(gzip.open(grammar_loc + "grammar.%d.gz"%line_counter, 'rb'))
        line_counter += 1
        for phrase in phrases:
            phrase_words = phrase.split()
            start, end = containsSequence(phrase_words, words)
            if start > -1 and end > -1:
                phrase_pos = [extract_training.collapsePOS(pos) for pos in pos_tags[start:end]]
                #currently doing bigrams only - need to figure out a way to combine representaitons for longer phrases
                phraseRep = model.computeComposedRep(phrase, ' '.join(phrase_pos))
                context = extractContext(words, start, end, numContext, model, noise)
                #comp = scoreLogReg(context, phrase, phraseRep, model, noise) 
                scoreSkipGram(context, phrase, phraseRep, model, noise)
                #print "Phrase '%s'; POS '%s'; %r"%(phrase, ' '.join(phrase_pos), comp)
     
if __name__ == "__main__":
    main()
