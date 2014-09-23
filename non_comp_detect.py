#!/usr/bin/python -tt

'''
File: non_comp_detect.py
Date: August 21, 2014
Description: this script reads in the parameters of compositional
combiner functions and word vectors, and for a given POS-tagged
corpus, computes phrasal representations for the phrases in the corpus.
It also compares these phrasal representations to a list of externally
computed "distances" between directly learned word2vec representations
for the phrases (the phrases can be output with the -v flag, and distances
to word2vec phrases can be computed with analysis/compute_distances.py),
and outputs the correlation with these distances. 
'''

import sys, commands, string, getopt, math, gzip, re
import numpy as np
import scipy as sp
import multiprocessing as mp
import extract_training
from compute_composed import *

class Unigrams:
    def __init__(self, filename, numStop):
        fh = open(filename, 'rb')
        counter = 0
        self.stopwords = []
        self.numStop = numStop
        self.unigrams = {}
        for line in fh:
            elements = line.strip().split('\t')
            self.unigrams[elements[0]] = float(elements[1])
            if counter < self.numStop and elements[0] != "<s>" and elements[0] != "</s>":
                self.stopwords.append(elements[0])
                counter += 1
        normalizer = sum(self.unigrams.values())
        for key, value in self.unigrams.items():
            self.unigrams[key] = value / normalizer
    
    def checkStopWord(self, word):
        return word in self.stopwords

'''
log likelihood-based scorer
'''
def scoreSkipGram(context, phraseRep, model):
    aggregate_score = 0.0
    for word in context:
        if word in model.contextVecs:
            wordRep = model.contextVecs[word]            
            aggregate_score += np.dot(wordRep, phraseRep)
    return aggregate_score

'''
cosine similarity-based scorer; does not take into account context
'''
def scoreCosineSim(phrase, phraseRep, model):
    words = phrase.split()
    wordVec1 = model.wordVecs[words[0]]
    wordVec2 = model.wordVecs[words[1]]
    cosSim1 = np.divide(np.dot(wordVec1, phraseRep), np.linalg.norm(wordVec1) * np.linalg.norm(phraseRep))
    cosSim2 = np.divide(np.dot(wordVec2, phraseRep), np.linalg.norm(wordVec2) * np.linalg.norm(phraseRep))
    return 0.5*(cosSim1 + cosSim2)

'''
macro for function below
'''
def checkArity(rule):
    return len(re.findall(r'\[([^]]+)\]', rule))

'''
used for extracting phrases that we want to score for a given sentence (arg: filehandle to .gz per-sentence grammar file)
'''
def processRules(grammar_fh):
    seen_rules = []
    preterm_rules = []
    for rule in grammar_fh: 
        src_rule = rule.strip().split(' ||| ')[1]
        if not src_rule in seen_rules:
            seen_rules.append(src_rule)
            if checkArity(src_rule) == 0 and len(src_rule.split()) == 2: #second condition is temporary; currently doing bigrams only 
                preterm_rules.append(src_rule)
    return preterm_rules

'''
extracts context from a sentence. Optionally filters for stop words in the context. 
'''
def extractContext(words, start_idx, end_idx, context_size, model, stopW):
    start = start_idx-1
    left_context = []
    while start > -1: #extract left context
        left_word = words[start]
        if left_word in model.contextVecs:
            if stopW.numStop > 0:
                if not stopW.checkStopWord(left_word):
                    left_context.append(left_word)
            else:
                left_context.append(left_word)
        if len(left_context) == context_size:
            break
        start -= 1
    end = end_idx
    right_context = []
    while end < len(words): #extract right context
        right_word = words[end]
        if right_word in model.contextVecs:
            if stopW.numStop > 0:
                if not stopW.checkStopWord(right_word):
                    right_context.append(right_word)
            else:
                right_context.append(right_word)
        if len(right_context) == context_size:
            break
        end += 1
    return left_context + right_context

'''
simple function to extract the indices of a subsequence given a sequence. 
This is used so that we can extract POS tags for a given word sequence
from a word-POS sentence. 
'''
def containsSequence(subseq, seq):
    for i in xrange(len(seq)-len(subseq)+1):
        for j in xrange(len(subseq)):
            if seq[i+j] != subseq[j]:
                break
        else:
            return i, i+len(subseq)
    return -1, -1

'''
this variant of the normalizer function is called when we are dealing with a thread
'''
def computeNormalizerThread(model, phrase_tuples, uniModel, uniCorrection, out_q): 
    revised_tuples = []
    for phrase, phrase_pos, context, phraseRep, score in phrase_tuples:
        normalizer = 0
        for word in model.contextVecs:
            contextVec = model.contextVecs[word]
            normalizer += math.exp(np.dot(contextVec, phraseRep))
        normalized_score = score - len(context)*math.log(normalizer)
        if uniCorrection:
            uniLogProb = 0
            for word in context: 
                if word in uniModel.unigrams:
                    uniLogProb += math.log(uniModel.unigrams[word])                
            normalized_score -= uniLogProb
        revised_tuples.append((phrase, phrase_pos, context, phraseRep, normalized_score))
    out_q.put(revised_tuples)

'''
controller/wrapper function for computeNormalizerThread
'''
def computeNormalizerParallel(phrase_tuples, numJobs, model, uniModel, uniCorrection):
    out_q = mp.Queue()
    procs = []
    revised_tuples = []
    chunksize = int(math.floor(len(phrase_tuples) / float(numJobs)))
    for proc in range(numJobs): 
        end = len(phrase_tuples) if proc == numJobs-1 else (proc+1)*chunksize
        tuples_proc = phrase_tuples[chunksize*proc:end]
        p = mp.Process(target=computeNormalizerThread, args=(model, tuples_proc, uniModel, uniCorrection, out_q))
        procs.append(p)
        p.start()
    for proc in range(numJobs):
        revised_tuples += out_q.get()
    for p in procs:
        p.join()
    return revised_tuples

'''
single-thread version of computing normalizer; deprecated
'''
def computeNormalizer(model, phraseRep, numJobs):
    normalizer = 0
    for word in model.contextVecs:
        contextVec = model.contextVecs[word]
        normalizer += math.exp(np.dot(contextVec, phraseRep))
    return normalizer

'''
function that goes through a corpus, extracts relevant phrases from the per-sentence grammars,
and scores each of those phrases in its appropriate context by first computing the phrasal
representation and then calling the scorer. 
The function can also be used to just print the relevant vectors. 
'''
def scorePhraseVectors(model, uniModel, numContext, grammar_loc, printOnly, cosine, headed): 
    line_counter = 0
    phrase_tuples = []
    for line in sys.stdin: #each line is a POS-tagged sentence (sequence of word#POS pairs)
        words, pos_tags = zip(*[word_pos.split('#') for word_pos in line.strip().split()])
        phrases = processRules(gzip.open(grammar_loc + "grammar.%d.gz"%line_counter, 'rb'))
        line_counter += 1
        for phrase in phrases:            
            if model.checkVocab(phrase): 
                phrase_words = phrase.split()
                start, end = containsSequence(phrase_words, words)
                if start > -1 and end > -1:
                    phrase_pos = [extract_training.collapsePOS(pos) for pos in pos_tags[start:end]]
                    phraseRep = model.computeComposedRep(phrase, ' '.join(phrase_pos))
                    if printOnly:
                        model.printVector('_'.join(phrase_words), phraseRep)
                    else:
                        context = extractContext(words, start, end, numContext, model, uniModel)
                        score = scoreCosineSim(phrase, phraseRep, model) if cosine else scoreSkipGram(context, phraseRep, model) 
                        if headed:
                            headedRep = model.computeHeadedRep(phrase, ' '.join(phrase_pos))
                            headedScore = scoreCosineSim(phrase, headedRep, model) if cosine else scoreSkipGram(context, headedRep, model)
                            score = headedScore
                        #if headedScore > score:
                        #        score = headedScore
                        phrase_tuples.append((phrase, ' '.join(phrase_pos), context, phraseRep, score))
    return phrase_tuples

'''
prints the relevant scores and distances out
'''
def printScoresAndDistances(revised_tuples, model, numContext, averaging, perplexity, distanceDict, printFullOnly, printPOSOnly):
    scores = []
    distances = []
    pos_scores_dist = {}
    for phrase, phrase_pos, context, phraseRep, score in revised_tuples:        
        numWordsInContext = sum([word in model.contextVecs for word in context])        
        if perplexity:
            score = math.exp(-score / numWordsInContext)
            scores.append(score)            
        elif averaging:
            score /= numWordsInContext
            scores.append(-score)
        else:
            scores.append(-score)
        distance = distanceDict[phrase] if phrase in distanceDict else -1
        distances.append(distance)
        pos_list = pos_scores_dist[phrase_pos] if phrase_pos in pos_scores_dist else []
        pos_list.append((scores[-1], distances[-1]))
        pos_scores_dist[phrase_pos] = pos_list
        if not printPOSOnly:
            if printFullOnly:
                if numWordsInContext == numContext:
                    print "%s\t%s\t%.3f\t%.3f \t%s"%(phrase, phrase_pos, score, distance, ' '.join(context))
            else:
                print "%s\t%s\t%.3f\t%.3f \t%s"%(phrase, phrase_pos, score, distance, ' '.join(context))
    return pos_scores_dist, scores, distances

def printPOSInfo(pos_scores_dist):
    for pos_pair in pos_scores_dist:
        pos_scores, pos_distances = zip(*pos_scores_dist[pos_pair])
        if len(pos_scores) > 1:
            pos_coeff, pos_pval = sp.stats.stats.pearsonr(pos_scores, pos_distances)
            print "%s\t%d\t%.3f\t%.3f"%(pos_pair, len(pos_scores), pos_coeff, sum(pos_distances) / len(pos_distances))

def main():    
    (opts, args) = getopt.getopt(sys.argv[1:], 'acCfhl:n:pPs:uv')
    concat = False
    numContext = 2
    numStop = 20
    averaging = False
    uniCorrection = False
    numJobs = -1
    perplexity = False #-P
    cosine = False
    headed = False
    printVecOnly = False #-v
    printFullOnly = False #-f
    printPOSOnly = False #-p
    for opt in opts:
        if opt[0] == '-a':
            averaging = True
        elif opt[0] == '-c':
            concat = True
        elif opt[0] == '-l':
            numContext = int(opt[1])
        elif opt[0] == '-s': #if numStop = 0, then no stop words used
            numStop = int(opt[1])
        elif opt[0] == '-u': #unigram correction
            uniCorrection = True
        elif opt[0] == '-n': #normalize the probabilities
            numJobs = int(opt[1]) #arg: number of processes to use
        elif opt[0] == '-P': #perplexity calculation
            perplexity = True
        elif opt[0] == '-C':
            cosine = True
        elif opt[0] == '-h':
            headed = True
        elif opt[0] == '-p':
            printPOSOnly = True
        elif opt[0] == '-v':
            printVecOnly = True
        elif opt[0] == '-f':
            printFullOnly = True
    model = CompoModel(args[2], concat, True)
    model.readVecFile(args[0], "word")
    model.readVecFile(args[1], "context")
    uniModel = Unigrams(args[3], numStop)
    grammar_loc = args[4]
    distance_fh = args[5]
    distanceDict = {}
    for line in open(distance_fh, 'rb'): #read in distances
        elements = line.strip().split('\t')
        distanceDict[elements[0]] = float(elements[1])
    if uniCorrection and numJobs < 0: 
        sys.stderr.write("Error! Unigram correction only valid if normalization used\n")
        sys.exit()
    if averaging and perplexity:
        sys.stderr.write("Error! Cannot do both averaging and perplexity score at the same time\n")
        sys.exit()
        
    phrase_tuples = scorePhraseVectors(model, uniModel, numContext, grammar_loc, printVecOnly, cosine, headed)
    if not printVecOnly:
        revised_tuples = computeNormalizerParallel(phrase_tuples, numJobs, model, uniModel, uniCorrection) if numJobs > 0 else phrase_tuples
        pos_score_dist, scores, distances = printScoresAndDistances(revised_tuples, model, numContext, averaging, perplexity, distanceDict, printFullOnly, printPOSOnly)
        if printPOSOnly:
            printPOSInfo(pos_score_dist)
        coeff, pval = sp.stats.stats.pearsonr(scores, distances)
        print "Out of %d samples, correlation between compositional model score and distance is %.3f (%.3f)"%(len(scores), coeff, pval)
        print "Average distance between directly learned representations and composed representations: %.3f"%(sum(distances) / len(distances))

if __name__ == "__main__":
    main()
