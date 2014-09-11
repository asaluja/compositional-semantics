#!/usr/bin/python -tt

import sys, commands, string
phrases = []

def checkPhrases(line):
    words = line.split()
    sentence = []
    if len(words) > 1:
        cur_idx = 1
        last_is_phrase = False
        while cur_idx < len(words):
            last_word = words[cur_idx-1]
            word = words[cur_idx]
            phrase = ' '.join([last_word, word])
            if phrase in phrases: #one of the n-grams that we need to replace
                sentence.append('_'.join([last_word, word]))
                last_is_phrase = True
                cur_idx += 2
            else:
                sentence.append(last_word)
                last_is_phrase = False
                cur_idx += 1
        else:
            if not last_is_phrase:
                sentence.append(words[-1])
        return sentence
    else:
        return words
                
def main():
    fh = open(sys.argv[1], 'rb')
    for line in fh:
        phrases.append(line.strip())
    for line in sys.stdin:
        words = checkPhrases(line.strip())
        print ' '.join(words)

if __name__ == "__main__":
    main()
