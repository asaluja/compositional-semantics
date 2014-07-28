#!/usr/bin/python -tt

'''
File: extract_training.py
Date: June 18, 2014
Description: this script takes as input the location of a PPDB database, 
filters out certain many-to-one/one-to-many pairs (corresponding to numerical
entries), POS tags the results, and selects particular pairs. 
Usage: 
'''

import sys, commands, string, getopt
from nltk.tag.stanford import POSTagger

POS_tagger = '/opt/tools/stanford-postagger-full-2013-11-12/stanford-postagger.jar'
POS_model = '/opt/tools/stanford-postagger-full-2013-11-12/models/wsj-0-18-bidirectional-nodistsim.tagger'

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'anv')
    selector = None
    for opt in opts:
        if opt[0] == '-a': #adjective-noun
            selector = "adj"
        elif opt[0] == '-d': #determiner
            selector = "det"
        elif opt[0] == '-n': #noun-noun
            selector = "noun"
        elif opt[0] == '-v': #verb-noun
            selector = "verb"
    if selector is None:
        sys.stderr.write("Error: need to provide one of the following flags to determine what word pairs to extract: -a (adjective-noun), -n (noun-noun), or -v (verb-noun)\n")
        sys.exit()

    training_tuples = set()
    for line in sys.stdin: #first, go through PPDB and select the first round of phrases
        elements = line.strip().split(' ||| ')
        if len(elements[1].split()) == 2 or len(elements[2].split()) == 2: #only look at 2-to-1 or 1-to-2 paraphrases
            many_phrase = elements[1] if len(elements[1].split()) == 2 else elements[2]
            one_phrase = elements[1] if len(elements[1].split()) == 1 else elements[2]
            isNumber = False
            for token in many_phrase.split():
                if is_number(token):
                    isNumber = True
            if not isNumber:
                training_tuples.add((one_phrase, many_phrase))
    tagger = POSTagger(POS_model, POS_tagger)
    for element in training_tuples:
        pos_tags = [word_pos[1] for word_pos in tagger.tag(element[1].split())] #extracts POS tags for multi-word phrase
        contains_noun = "NN" in pos_tags or "NNS" in pos_tags or "NNP" in pos_tags or "NNPS" in pos_tags
        if contains_noun:
            valid = False
            if selector == "adj":
                valid = "JJ" in pos_tags or "JJR" in pos_tags or "JJS" in pos_tags
            elif selector == "verb":
                valid = "VB" in pos_tags or "VBD" in pos_tags or "VBG" in pos_tags or "VBN" in pos_tags or "VBP" in pos_tags or "VBZ" in pos_tags
            elif selector == "det":
                valid = "DT" in pos_tags
            elif selector == "noun":
                valid = True #start off with valid being true
                for pos_tag in pos_tags:
                    valid = valid & (pos_tag == "NN" or pos_tag == "NNS" or pos_tag == "NNP" or pos_tag == "NNPS")
            else:
                sys.stderr.write("Could not regonize selector\n")
            if valid:
                print "%s ||| %s"%(element[0], element[1])

if __name__ == "__main__":
    main()
