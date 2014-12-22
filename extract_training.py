#!/usr/bin/python -tt

'''
File: extract_training.py
Date: June 18, 2014
Description: this script takes as input the location of a PPDB database, 
filters out certain many-to-one/one-to-many pairs (corresponding to numerical
entries), POS tags the results, and selects particular pairs. 
Update (August 21, 2014): repurposed this file to make it contain the 'TrainingExtractor' class. 
It can either be run directly, in which case it goes through PPDB, POS tags the various
examples, and selects examples based on the POS pair requested, or it can be run by directly
interfacing with the class, e.g., via learning.py
'''

import sys, commands, string, getopt
from nltk.tag.stanford import POSTagger

def collapsePOS(pos):
    new_pos = ""
    if pos == "JJR" or pos == "JJS":
        new_pos = "JJ"
    elif pos == "NNS" or pos == "NNP" or pos == "NNPS":
        new_pos = "NN"
    elif pos == "VBD" or pos == "VBG" or pos == "VBN" or pos == "VBP" or pos == "VBZ":
        new_pos = "VB"
    elif pos == "RBR" or pos == "RBS":
        new_pos = "RB"
    else:
        new_pos = pos
    return new_pos

class TrainingExtractor:
    TAGGER_LOC = '/opt/tools/stanford-postagger-full-2013-11-12/stanford-postagger.jar'
    TAGGER_MODEL = '/opt/tools/stanford-postagger-full-2013-11-12/models/wsj-0-18-bidirectional-nodistsim.tagger'

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def __init__(self, database_loc, selector = "all", top_combiners = 3, POS = False, filter_number = True): 
        self.database_loc = database_loc
        self.selector = selector
        self.top_combiners = top_combiners
        self.filter_number = filter_number
        self.training_examples = {}
        self.pos_provided = POS

    def extract_examples(self):
        training_tuples = set()
        db_fh = open(self.database_loc, 'rb')
        for line in db_fh: #going through PPDB
            elements = line.strip().split(' ||| ')
            if len(elements[1].split()) == 2 or len(elements[2].split()) == 2: #only look at 2-to-1 or 1-to-2 paraphrases
                many_phrase = elements[1] if len(elements[1].split()) == 2 else elements[2]
                one_phrase = elements[1] if len(elements[1].split()) == 1 else elements[2]
                if self.filter_number:
                    isNumber = False
                    for token in many_phrase.split():
                        if self.pos_provided:
                            token = token.split('#')[0]
                        if self.is_number(token):
                            isNumber = True
                    if not isNumber:
                        training_tuples.add((one_phrase, many_phrase))
                else:
                    training_tuples.add((one_phrase, many_phrase))
        tagger = POSTagger(self.TAGGER_MODEL, self.TAGGER_LOC)
        self.training_examples = {} #reset training examples
        for element in training_tuples: #now, tag the resulting data
            words = element[1].split()
            words_only = ""
            if self.pos_provided:
                words_only = ' '.join([word_pos.split('#')[0] for word_pos in words])
            pos_tags = [word_pos.split('#')[1] for word_pos in words] if self.pos_provided else [word_pos[1] for word_pos in tagger.tag(words)]            
            #pos_tags = [word_pos[1] for word_pos in tagger.tag(words)]
            collapsed_pos = []
            for pos in pos_tags: #cluster certain pos tags together
                new_pos = collapsePOS(pos)
                collapsed_pos.append(new_pos)
            key = ' '.join(collapsed_pos)
            examples = self.training_examples[key] if key in self.training_examples else []
            if self.pos_provided:
                examples.append(' '.join([element[0], words_only]))
            else:
                examples.append(' '.join([element[0], element[1]]))
            self.training_examples[key] = examples
        sys.stderr.write("PPDB training data tagged and sorted\n")
        db_fh.close()

    def print_examples(self):
        if self.selector == "all":
            counter = 0
            for k in sorted(self.training_examples, key=lambda k: len(self.training_examples[k]), reverse=True):
                examples = ' ||| '.join(self.training_examples[k])
                key = k if counter < self.top_combiners else "X X"
                counter += 1
                print "%s ||| %s"%(key, examples)
        else:
            pair = ""
            if self.selector == "det":
                pair = "DT NN"
            elif self.selector == "adj":
                pair = "JJ NN"
            elif self.selector == "noun":
                pair = "NN NN"
            elif self.selector == "verb":
                pair = "VB NN"
            else:
                sys.stderr.write("Error: given selector is invalid, so cannot print examples\n")
                sys.exit()
            examples = " ||| ".join(self.training_examples[pair])
            print "%s ||| %s"%(pair, examples)
            
    def package_examples(examples, paraphrases, key):
        for paraphrase in paraphrases:
            elements = paraphrase.split()
            one_phrase = elements[0]
            many_phrase = ' '.join(elements[1:])
            examples.append((key, one_phrase, many_phrase))

    def return_examples(self):
        examples = []
        if self.selector == "all":
            counter = 0
            for k in sorted(self.training_examples, key=lambda k: len(self.training_examples[k]), reverse=True):
                key = k if counter < self.top_combiners else "X X"
                package_examples(examples, self.training_examples[k], key)
                counter += 1
            return examples
        else:
            pair = ""
            if self.selector == "det":
                pair = "DT NN"
            elif self.selector == "adj":
                pair = "JJ NN"
            elif self.selector == "noun":
                pair = "NN NN"
            elif self.selector == "verb":
                pair = "VB NN"
            else:
                sys.stderr.write("Error: given selector is invalid, so cannot print examples\n")
                sys.exit()
            package_examples(examples, self.training_examples[pair], pair)
            return examples

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'aA:npv')
    selector = None
    POS = False
    top_combiners = -1
    for opt in opts:
        if opt[0] == '-a': #adjective-noun
            selector = "adj"
        elif opt[0] == '-n': #noun-noun
            selector = "noun"
        elif opt[0] == '-v': #verb-noun
            selector = "verb"
        elif opt[0] == '-A': #all
            selector = "all"
            top_combiners = int(opt[1])
        elif opt[0] == '-p': 
            POS = True
    if selector is None:
        sys.stderr.write("Error: need to provide one of the following flags to determine what word pairs to extract: -a (adjective-noun), -n (noun-noun), -v (verb-noun), or -A (all, with argument on number of POS combiners\n")
        sys.exit()
    dbloc = args[0]
    extractor = TrainingExtractor(dbloc, selector, top_combiners, POS)
    extractor.extract_examples()
    extractor.print_examples()

if __name__ == "__main__":
    main()
