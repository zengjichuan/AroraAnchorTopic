__author__ = 'admin'
import os
import time
import csv
def trim_vacab(vocab, docs, words, Ddw):
    pass

def word_freqs(words, Ddw):
    pass

def word_idfs(words, docs):
    pass

def load_uci(basename, min_tf=0, min_tfidf=0):
    vocab = os.path.join('uci', 'vocab.%s.txt' % basename)
    docword = os.path.join('uci', 'docword.%s.txt' % basename)
    print "-- Reading files"
    startT = time.time()
    word = [w for (w,) in csv.reader(open(vocab))]

    print "--- %s seconds ---" % (time.time() - startT)