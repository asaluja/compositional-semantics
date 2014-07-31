#!/usr/bin/bash -e

scriptsLoc=$1
workingLoc=$2
wordVecs=$3

python ${scriptsLoc}/learning.py -n -j 8 -r lars $wordVecs ${workingLoc}/adj_nn.params < ${workingLoc}/adj_nn.training
python ${scriptsLoc}/learning.py -n -j 8 -r lars $wordVecs ${workingLoc}/nn_nn.params < ${workingLoc}/nn_nn.training

python ${scriptsLoc}/compute_composed.py -a $wordVecs ${workingLoc}/adj_nn.params ${workingLoc}/nn_nn.params < ${workingLoc}/all.bigrams.tagged > ${workingLoc}/additive.bigram.reps
python ${scriptsLoc}/compute_composed.py -m $wordVecs ${workingLoc}/adj_nn.params ${workingLoc}/nn_nn.params < ${workingLoc}/all.bigrams.tagged > ${workingLoc}/multiplicative.bigram.reps
python ${scriptsLoc}/compute_composed.py $wordVecs ${workingLoc}/adj_nn.params ${workingLoc}/nn_nn.params < ${workingLoc}/all.bigrams.tagged > ${workingLoc}/learned.bigram.reps

