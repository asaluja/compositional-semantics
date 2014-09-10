`compositional-semantics` is a package for learning compositional functions given vector representations of words.  A special feature is the detection and handling of non-compositional phrases.  

## System Requirements

- NumPy and SciPy

## End-to-end Pipeline Instructions

Inputs: word vectors, context vectors, PPDB
Note: 

1. In order to make things faster, it is better to run the TrainingExtractor module *a priori* on PPDB: 

```
python extract_training.py -A X ppdb_loc > sorted_ppdb_examples
```

The `-A X` flag indicates we should output all POS pairs: specific pairs for the top X, and the rest in a generic 'X X' pair type. Other options available are:
  - `-a`: adjective-noun
  - `-n`: noun-noun
  - `-d`: determiner-noun
  - `-v`: verb-noun

2. Run the learning module to learn and output the compositional functions:

```
python learning.py wordVecs sorted_ppdb_examples parameters
```

Flags:
  - `-n`: normalize word vectors to be unit norm (recommended)
  - `-j X`: sets number of jobs/cores. Default is 4
  - `-f`: filter duplicates; this removes training examples where a word is repeated ('trade union' --> 'union')
  - `-r X`: sets the regressor to be used when learning the compositional functions. Default is lasso
  - `-d`: diagonal approximation: only the diagonal of the resulting tensor parameters is active
  - `-c`: concatenative model. Instead of N^2 covariates, where N is the dimension of one vector, we have 2*N covariates. Note that the diagonal approximation and the concatenative model flags cannot be used together. 
  - `-p`: use PPDB; this means that the second argument passed to the script is actually PPDB, not the POS-sorted examples

3. If you want to evaluate the learned representations against the human evaluations (in the Mitchell & Lapata 2010 study), you can run the evaluation module:

```
python evaluation.py wordVecs parameters < mitchell_lapata_scores
```

Flags:
  - `-n`: normalize word vectors to be unit norm (recommended)
  - `-j`: adjective-noun evaluation
  - `-N`: noun-noun evaluation
  - `-m`: output correlation scores from a simple point-wise multiplicative model (no learning, so parameters argument is dummy)
  - `-d`: output most divergent and least divergent similarity pairs between learned representations and human evaluations
  - `-c`: concatenative model. If the parameters have been learned using a concatenative model, then this flag is required. 
  - `-e`: compute Spearman's correlation scores only with the most extreme similarities on the 1-7 scale

4. If you want to get non-compositionality scores for phrases using these parameters:

```
python non_comp_detect.py wordVecs contextVecs parameters unigram_dev_counts unigram_training_counts per_sentence_grammar_location
```

Flags:
  - `-c`: concatenative model. If the parameters have been learned using a concatenative model, then this flag is required. 
  - `-C X`: number of context words on either side to consider in the window (total window size: X * 2); default is 1
  - `-s X`: number of negative samples (in the process of being removed)

## Things to add

- include version of word2vec where we can output context vectors as well in repository
- stand-alone interface to CompoModel module where we output the phrasal representations given a list of phrases
