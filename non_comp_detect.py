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
def scoreCosineSim(phrase, phraseRep, model, alpha):
    words = phrase.split()
    avgCosSim = 0
    if len(words) > 2: #then equal weighting
        for word in words:
            wordVec = model.wordVecs[word]
            cosSim = np.divide(np.dot(wordVec, phraseRep), np.linalg.norm(wordVec) * np.linalg.norm(phraseRep))
            avgCosSim += cosSim
        return avgCosSim / len(words)
    else: #if length 2, then can put variable weights
        wordVec1 = model.wordVecs[words[0]]
        wordVec2 = model.wordVecs[words[1]]
        sim1 = np.divide(np.dot(wordVec1, phraseRep), np.linalg.norm(wordVec1) * np.linalg.norm(phraseRep))
        sim2 = np.divide(np.dot(wordVec2, phraseRep), np.linalg.norm(wordVec2) * np.linalg.norm(phraseRep))
        avgCosSim = alpha*sim1 + (1-alpha)*sim2
        return avgCosSim

'''
macro for function below
'''
def checkArity(rule):
    return len(re.findall(r'\[([^]]+)\]', rule))

'''
used for extracting phrases that we want to score for a given sentence (arg: filehandle to .gz per-sentence grammar file)
'''
def processRules(grammar_fh, distanceCorr):
    seen_rules = []
    preterm_rules = []
    for rule in grammar_fh: 
        src_rule = rule.strip().split(' ||| ')[1]
        if not src_rule in seen_rules:
            seen_rules.append(src_rule)
            if not distanceCorr:
                if checkArity(src_rule) == 0 and len(src_rule.split()) > 1:
                    preterm_rules.append(src_rule)
            else: #if doing distanceCorr, then only consider bigrams
                if checkArity(src_rule) == 0 and len(src_rule.split()) == 2:
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
    for phrase, phrase_pos, context, phraseRep, score, sentNum in phrase_tuples:
        normalizer = 0
        for word in model.contextVecs: #guaranteed that word is in context because this criterion is checked in extractContext
            contextVec = model.contextVecs[word]
            normalizer += math.exp(np.dot(contextVec, phraseRep))
        normalized_score = score - len(context)*math.log(normalizer)
        if uniCorrection:
            uniLogProb = 0
            for word in context: 
                if word in uniModel.unigrams:
                    uniLogProb += math.log(uniModel.unigrams[word])                
            normalized_score -= uniLogProb
        revised_tuples.append((phrase, phrase_pos, context, phraseRep, normalized_score, sentNum))
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
def scoreSegmentations(model, uniModel, numContext, grammar_loc, printOnly, cosine, distanceCorr, writePSG): 
    line_counter = 0
    phrase_tuples = []
    for line in sys.stdin: #each line is a POS-tagged sentence (sequence of word#POS pairs) 
        phrase_context = None
        if not writePSG:
            phrase_context = line.strip().split(' ||| ')
        phrases = processRules(gzip.open(grammar_loc + "grammar.%d.gz"%line_counter, 'rb'), distanceCorr) if writePSG else [phrase_context[0]]
        words, pos_tags = zip(*[word_pos.split('#') for word_pos in line.strip().split()]) if writePSG else zip(*[word_pos.split('#') for word_pos in phrase_context[1].split()])
        for phrase in phrases: #phrase can have NTs in it
            if model.checkVocab(phrase):                 
                phrase_words = phrase.split()
                start, end = containsSequence(phrase_words, words) #if we can get this information from the PSG, then it would be much easier
                if start > -1 and end > -1:
                    phrase_pos = [extract_training.collapsePOS(pos) for pos in pos_tags[start:end]]                    
                    phraseRep = model.computeComposedRep(phrase, ' '.join(phrase_pos))
                    if printOnly:
                        model.printVector('_'.join(phrase_words), phraseRep)
                    else:
                        context = extractContext(words, start, end, numContext, model, uniModel)
                        score = scoreCosineSim(phrase, phraseRep, model, cosine) if cosine >= 0 else scoreSkipGram(context, phraseRep, model) 
                        phrase_tuples.append((phrase, ' '.join(phrase_pos), context, phraseRep, score, line_counter))
        line_counter += 1
    return phrase_tuples

'''
prints the relevant scores and distances out
'''
def printScoresAndDistances(revised_tuples, model, numContext, averaging, perplexity, distanceDict, printFullOnly, printPOSOnly):
    scores = []
    distances = []
    pos_scores_dist = {}
    for phrase, phrase_pos, context, phraseRep, score, sentNum in revised_tuples:
        numWordsInContext = len(context)
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

def writePerSentenceGrammar(loc_in, loc_out, phrase_tuples, printComposed, averaging, perplexity, binning, numBins, featNT):    
    sentDict = {}
    vecSize = len(phrase_tuples[0][3])
    for phrase, phrase_pos, context, phraseRep, score, sentNum in phrase_tuples:
        phraseDict = sentDict[sentNum] if sentNum in sentDict else {}
        if len(context) > 0:
            if perplexity:
                score = math.exp(-score / len(context))
                score /= 100000 #just for feature scaling purposes
            elif averaging:
                score /= len(context)
        phraseDict[phrase] = score if not printComposed else (score, phraseRep) 
        sentDict[sentNum] = phraseDict

    if binning != "":
        for sentNum in sentDict:
            phraseDict = sentDict[sentNum]
            phrase_scores = phraseDict.items()
            scores = [key_val[1] for key_val in phrase_scores] if not printComposed else [key_val[1][1] for key_val in phrase_scores]
            bins = []
            if binning == "width": #compute the bin start and end values
                histo, bins = np.histogram(scores, numBins)
            else: #can only be size since binning != ""
                numElementsPerBin = float(len(scores)) / numBins
                probs = np.array(range(numBins+1))*numElementsPerBin
                probs /= float(len(scores))
                bins = sp.stats.mstats.mquantiles(scores, probs)
            binned_scores = np.digitize(scores, bins)
            for idx, key in enumerate(phrase_scores): #bin the values
                phrase = key[0]
                score = binned_scores[idx]
                if score > numBins:
                    score = numBins
                phraseDict[phrase] = score if not printComposed else (score, key[1][1]) 
            sentDict[sentNum] = phraseDict

    numSentences = max(sentDict.keys())
    for line_counter in xrange(numSentences+1):
        grammar_fh = gzip.open(loc_in+"grammar.%d.gz"%line_counter, 'rb')
        out_fh = gzip.open(loc_out+"grammar.%d.gz"%line_counter, 'w')
        if line_counter not in sentDict: #to handle sentences where we do not extract rule with any phrase longer than length 1
            for rule in grammar_fh: 
                elements = rule.strip().split(' ||| ')
                features = elements[3]
                features += " NoSeg=1"
                arrayToPrint = elements[:3] + [features] + elements[4:]
                lineToPrint = ' ||| '.join(arrayToPrint)
                out_fh.write("%s\n"%lineToPrint)
            grammar_fh.close()
            out_fh.close()
        else:
            phraseDict = sentDict[line_counter]
            for rule in grammar_fh:
                elements = rule.strip().split(' ||| ')
                src_rule = elements[1]
                features = elements[3]
                if featNT: #featurize NTs
                    subPhrases = re.split(r'\[(?:[^]]+)\]', src_rule) #split rule into lexical items divided by NTs
                    score = 0
                    counter = 0
                    composedRep = np.zeros(vecSize)
                    for subphrase in subPhrases: 
                        if subphrase.strip() in phraseDict: #if the subphrase has a segmentation score, i.e., it is also a pre-terminal
                            counter += 1
                            score += phraseDict[subphrase.strip()] if not printComposed else phraseDict[subphrase.strip()][0]
                            if printComposed:
                                composedRep += phraseDict[subphrase.strip()][1]
                    if counter > 0: #valid segmentation score
                        features += " SegScore=%.3f SegOn=1"%(score / counter) #average over scores for each phrase in rule                        
                        #features += " SegOn=1" #for baseline purposes
                        if printComposed:
                            composedRep = np.divide(composedRep, counter)
                            for featName, featVal in enumerate(composedRep):
                                features += " Dimension%d=%.3f"%(featName, featVal)
                    else:
                        features += " NoSeg=1"
                elif src_rule in phraseDict:
                    if printComposed:
                        features += " SegScore=%.3f SegOn=1"%phraseDict[src_rule][0]
                        composedRep = phraseDict[src_rule][1]
                        for featName, featVal in enumerate(composedRep):
                            features += " Dimension%d=%.3f"%(featName, featVal)
                    else:
                        features += " SegScore=%.3f SegOn=1"%phraseDict[src_rule]
                else:
                    features += " NoSeg=1"                
                arrayToPrint = elements[:3] + [features] + elements[4:]
                lineToPrint = ' ||| '.join(arrayToPrint)
                out_fh.write("%s\n"%lineToPrint)
            grammar_fh.close()
            out_fh.close()

def writeNonCompScores(phrase_tuples, averaging, perplexity, binning, numBins):
    phraseDict = {}
    for phrase, phrase_pos, context, phraseRep, score, sentNum in phrase_tuples: #restructure into more suitable output
        if len(context) > 0:
            if perplexity:
                score = math.exp(-score / len(context))
            elif averaging:
                score /= len(context)
        scores = phraseDict[phrase] if phrase in phraseDict else []
        scores.append(score)
        phraseDict[phrase] = scores

    if binning != "": #for binning
        phrases = [] #for ordering purposes
        scores = [] 
        binned_phraseDict = {}
        for phrase in phraseDict: #reformat data
            scores_per_phrase = phraseDict[phrase]
            for score in scores_per_phrase:
                phrases.append(phrase)
                scores.append(score)
        bins = []
        if binning == "width":
            histo, bins = np.histogram(scores, numBins)
        else: #can only be size since binning != ""
            numElementsPerBin = float(len(scores)) / numBins
            probs = np.array(range(numBins+1))*numElementsPerBin
            probs /= float(len(scores))
            bins = sp.stats.mstats.mquantiles(scores, probs)
        binned_scores = np.digitize(scores, bins)
        for idx, phrase in enumerate(phrases):
            score = binned_scores[idx]
            if score > numBins: #should only happen once, to max(scores)
                score = numBins
            scores_per_phrase = binned_phraseDict[phrase] if phrase in binned_phraseDict else []
            scores_per_phrase.append(score)
            binned_phraseDict[phrase] = scores_per_phrase
        phraseDict = binned_phraseDict

    for phrase in phraseDict:
        scores = phraseDict[phrase]
        averageScore = sum(scores) / len(scores)
        print "%s ||| %.3f"%(phrase, averageScore)
        
def main():    
    (opts, args) = getopt.getopt(sys.argv[1:], 'aAb:B:cC:d:fhl:Mn:NpPrs:uvVw:')
    concat = False
    rightBranch = False
    numContext = 2
    numStop = 20
    averaging = False
    binning = ""
    numBins = 10 #default number of bins
    uniCorrection = False
    numJobs = -1
    perplexity = False #-P
    cosine = -1
    headed = False
    featNTs = False
    distanceCorr = ""
    printVecOnly = False #-v
    printFullOnly = False #-f
    printPOSOnly = False #-p
    printComposedRep = False
    addModel = False
    multModel = False
    writePSG = ""
    for opt in opts:
        if opt[0] == '-a':
            averaging = True
        elif opt[0] == '-b': #enable binning; argument is the type of bin
            binning = opt[1]
        elif opt[0] == '-B': #can change bin size with this option
            numBins = int(opt[1])
        elif opt[0] == '-c':
            concat = True
        elif opt[0] == '-r':
            rightBranch = True
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
            cosine = float(opt[1])
        elif opt[0] == '-h':
            headed = True
        elif opt[0] == '-d':
            distanceCorr = opt[1]
        elif opt[0] == '-p':
            printPOSOnly = True
        elif opt[0] == '-v':
            printVecOnly = True
        elif opt[0] == '-f':
            printFullOnly = True
        elif opt[0] == '-w':
            writePSG = opt[1]
        elif opt[0] == '-V':
            printComposedRep = True
        elif opt[0] == '-N': #featurize phrases with NTs
            featNTs = True
        elif opt[0] == '-M': #multiplicative model
            multModel = True
        elif opt[0] == '-A': #simple additive model
            addModel = True
    model = CompoModel(args[2], concat, True, headed, rightBranch, multModel, addModel)
    model.readVecFile(args[0], "word")
    model.readVecFile(args[1], "context")
    uniModel = Unigrams(args[3], numStop)
    grammar_loc_in = args[4]
    distanceDict = {}
    if distanceCorr != "": 
        for line in open(distanceCorr, 'rb'): #read in distances
            elements = line.strip().split('\t')
            distanceDict[elements[0]] = float(elements[1])
    if uniCorrection and numJobs < 0: 
        sys.stderr.write("Error! Unigram correction only valid if normalization used\n")
        sys.exit()
    if averaging and perplexity:
        sys.stderr.write("Error! Cannot do both averaging and perplexity score at the same time\n")
        sys.exit()
    if distanceCorr != "" and writePSG != "": 
        sys.stderr.write("Error! Cannot do both distance correlation computation and writing per-sentence grammar; disable one\n")
        sys.exit()
    if binning != "" and not (binning == "width" or binning == "size"):
        sys.stderr.write("Error! Argument to '-b' option needs to be either 'width' or 'size'\n")
        sys.exit()
    if printComposedRep and writePSG == "": 
        sys.stderr.write("Note: cannot print composed representation of phrase when not writing out per-sentence grammar; ignored\n")
    
    phrase_tuples = scoreSegmentations(model, uniModel, numContext, grammar_loc_in, printVecOnly, cosine, distanceCorr != "", writePSG != "")
    sys.stderr.write("Scored phrases in context\n")
    if not printVecOnly:
        revised_tuples = computeNormalizerParallel(phrase_tuples, numJobs, model, uniModel, uniCorrection) if numJobs > 0 else phrase_tuples
        sys.stderr.write("Normalized phrase scores (if requested)\n")
        if distanceCorr != "":
            pos_score_dist, scores, distances = printScoresAndDistances(revised_tuples, model, numContext, averaging, perplexity, distanceDict, printFullOnly, printPOSOnly)
            if printPOSOnly:
                printPOSInfo(pos_score_dist)
            coeff, pval = sp.stats.stats.pearsonr(scores, distances)
            print "Out of %d samples, correlation between compositional model score and distance is %.3f (%.3f)"%(len(scores), coeff, pval)
            print "Average distance between directly learned representations and composed representations: %.3f"%(sum(distances) / len(distances))
        if writePSG != "":
            writePerSentenceGrammar(grammar_loc_in, writePSG, revised_tuples, printComposedRep, averaging, perplexity, binning, numBins, featNTs)
            sys.stderr.write("Wrote per-sentence grammars\n")
        else: #then just write out phrase ||| average score over context sentences
            writeNonCompScores(revised_tuples, averaging, perplexity, binning, numBins)

if __name__ == "__main__":
    main()
