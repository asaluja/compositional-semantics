`compositional-semantics` is a package for learning compositional functions given vector representations of words.  A special feature is the detection and handling of non-compositional phrases.  

## System Requirements

- NumPy and SciPy

## End-to-end Pipeline Instructions

Inputs: word vectors, context vectors, PPDB

1. In order to make things faster, it is better to run the TrainingExtractor module *a priori* on PPDB: 

```
python extract_training.py -A X ppdb_loc > sorted_ppdb_examples
```

  The `-A X` flag indicates we should output all POS pairs: specific pairs for the top X, and the rest in a generic 'X X' pair type. Other options available are:
  - `-a`: adjective-noun
  - `-n`: noun-noun
  - `-d`: determiner-noun
  - `-v`: verb-noun
  - `-p`: POS tags have also been provided along with PPDB data

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
  - `-P`: linguistic regularization prior strength
  - `-m`: multivariate version of whatever regressor is chosen with `-r`. By default, we distribute estimation over cores, but with `-m` it's a single core process, but you get better R^2

3. If you want to evaluate the learned representations against the human evaluations (in the Mitchell & Lapata 2010 study), you can run the evaluation module:

```
python evaluation.py wordVecs parameters < mitchell_lapata_scores
```

  Flags:
  - `-j`: adjective-noun phrase similarity evaluation
  - `-J`: adjective-noun non-compositionality evaluation
  - `-n`: noun-noun phrase similarity evaluation
  - `-N`: noun-noun non-compositionality evaluation
  - `-m`: output correlation scores from a simple point-wise multiplicative model (no learning, so parameters argument is dummy)
  - `-a`: output correlation scores from a simple point-wise additive model (no learning, so parameters argument is dummy)
  - `-d`: output most divergent and least divergent similarity pairs between learned representations and human evaluations
  - `-c`: concatenative model. If the parameters have been learned using a concatenative model, then this flag is required. 
  - `-e`: compute Spearman's correlation scores only with the most extreme similarities on the 1-7 scale

By default, input vectors are assumed normalized. 

4. If you want to get non-compositionality scores for phrases using these parameters:

```
python non_comp_detect.py wordVecs contextVecs parameters unigram_counts per_sentence_grammar_loc_in < dev_corpus_with_pos_tags
```

  Flags:
  - `-c`: concatenative model. If the parameters have been learned using a concatenative model, then this flag is required. 
  - `-b X`: bin the values; argument is the type of binning (i.e., constant size bins or constant width bins); adds a type of non-linearity and more discrimination
  - `-B X`: number of bins; default: 10
  - `-a`: averaging; divide the log probability by the number of words to normalize for context length (beginning/end sentence effects)
  - `-l X`: context length; the number of words to look at on one side of the phrase (so the window is double this)
  - `-s X`: number of stop words to filter in the context. 0 means do not filter stop words. 
  - `-n X`: apply normalization (and use X cores in the process).  Not using this flag will leave scores as unnormalized. 
  - `-u`: apply unigram correction; this is a heuristic that divides the score, which is the probability of the context given the phrasal representation, by the product of the unigram probabilities of the context. Only works if the `-n` flag is enabled.   
  - `-P`: output perplexity instead of log probability; this also takes into account the length of the context
  - `-C`: score non-compositionality using a simple cosine similarity heuristic (ignoring context), where we average the cosine similarity of the phrasal representation with each of the word representations. 
  - `-v`: print phrasal representation vectors only; useful output if you want to compute the distances between these representations and directly learned word2vec phrasal representations. 
  - `-V`: print phrasal representation vectors when writing out per-sentence grammars as extra features in the MT grammar
  - `-f`: print and score phrases with full i.e., 2 times context length words in the context. 
  - `-p`: print correlation and distance information by POS pair only
  - `-d X`: compute correlation with distance between directly learned phrasal representation and composed representation; only works for bigrams. `X` denotes the location of the file with the distances pre-computed. 
  - `-r`: select right-branching when computing compositional representations for phrases of length greater than 2.  The default is left-branching. 
  - `-h`: impose heuristic headedness when computing compositional representations.  For 'JJ-NN' and 'NN-NN' the right word is the head word and only its representation is used; for 'VV-NN' the verb is the head word. 
  - `-w X`: write per-sentence grammars out; the grammars are in the same format as the grammars being read in, except we add an extra 'Segmentation' feature based on the non-compositionality score
  - `-N`: featurize rules witih non-terminals as well, based on the lexical items they contain. 
  - `-A`: evaluate non-compositionality using simple pointwise additive compositional representations. 
  - `-M`: evaluate non-compositionality using simple pointwise multiplicative compositional representations. 

  1. Note that in order to do non-compositionality detection, an evaluation set must be provided with the POS tags (separated by '#').  In order to do this, run the following command:

  ```
  java -mx5000m -cp $POSTAGGER/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model $POSTAGGER/models/english-bidirectional-distsim.tagger -textFile eval.lc-tok -outputFormat slashTags -tagSeparator \# -tokenize false > eval.lc-tok.pos
  ```

## Things to add

- include version of word2vec where we can output context vectors as well in repository (context vectors are necessary for non-compositionality calculation)
- stand-alone interface to CompoModel module where we output the phrasal representations given a list of phrases
