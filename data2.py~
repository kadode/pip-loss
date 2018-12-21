import os
import torch
from nltk import word_tokenize
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.cooccur2count = {}
        self.word2count = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.window = 5
        self.tokenize(path)
        
    def skipgram(self, sentence, i):
        inword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return inword, left + right

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                #words = word_tokenize(line)
                words = line.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    
        self.num_vocab = len(self.dictionary)
        print("Vocabulary : {}".format(self.num_vocab))
        print("Token : {}".format(tokens))
        self.M1 = np.zeros((self.num_vocab,self.num_vocab))
        self.M2 = np.zeros((self.num_vocab,self.num_vocab))
        half = int(tokens/2)
        # Tokenize file content
        with open(path, 'r') as f:
            #ids = torch.LongTensor(tokens)
            token = 0
            for j,line in enumerate(f):
                #sent = word_tokenize(line)
                sent = line.split()
                for i in range(len(sent)):
                    inword,contexts = self.skipgram(sent,i)
                    token += 1
                    for cxt in contexts:
                        if token < half:
                            self.M1[self.dictionary.word2idx[inword]][self.dictionary.word2idx[cxt]] += 1.0
                        else:
                            self.M2[self.dictionary.word2idx[inword]][self.dictionary.word2idx[cxt]] += 1.0

