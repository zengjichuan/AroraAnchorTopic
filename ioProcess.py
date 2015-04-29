__author__ = 'admin'
import os
import time
import csv
import numpy as np
import scipy.sparse as ss

def compact_doc_matrix(vocab, docs, words, V):
    # reindex docs
    nd = doc_lengths(docs, V)
    active = np.nonzero(nd > 0)
    map_doc_id = np.zeros(nd.size)
    map_doc_id[active] = range(active[0].size)
    docs = map_doc_id[docs]
    # reindex words
    nw = word_freqs(words, V)
    active = np.nonzero(nw > 0)
    map_word_id = np.zeros(nw.size)
    map_word_id[active] = range(active[0].size)
    words = map_word_id[words]
    # discard unused word from the vocabulary
    vocab = [vocab[i] for i in active[0]]
    m = len(set(doc for doc in docs))
    n = len(vocab)
    Ddw = ss.csr_matrix((V, (docs, words)), shape=(m,n))
    return vocab, Ddw

# X_(i,j) denotes doc i in word j
def compute_Q(X):
    nd = X.sum(1)

    # Compute the matrices and sum across documents to get Hhat
    scaling = np.multiply(nd, nd - 1)
    Hdt = X / scaling
    Hdtd = Hdt.sum(0)

    scaling = np.sqrt(np.multiply(nd, nd - 1))
    Hdw = X / scaling
    Q = Hdw.T * Hdw - ss.diags(np.squeeze(np.asarray(Hdtd)), 0)
    Q = ss.csc_matrix(Q / Q.sum(1))
    return Q

def trim_vacab(vocab, docs, words, V, min_tf=0, min_tfidf=0):
    tf = word_freqs(words, V)
    idf = word_idfs(words, docs)
    keep = np.logical_and(tf > min_tf, tf * idf > min_tfidf)
    #I = np.intersect1d(np.nonzero(tf > min_tf), np.nonzero(tf * idf > min_tfidf))
    I = np.nonzero(keep[words] == True)
    print "Trim: nwords=", len(vocab), "-> nwords=", np.count_nonzero(keep == True)
    print "Trim: nnz=", V.size, "-> nnz=", I[0].size
    # Compact the term-doc matrix and return
    return compact_doc_matrix(vocab, docs[I], words[I], V[I])

def doc_lengths(docs, V):
    counts = np.zeros(np.max(docs) + 1)
    for k in range(docs.size):
        counts[docs[k]] = counts[docs[k]] + V[k]
    return counts

def word_freqs(words, V):
    counts = np.zeros(np.max(words) + 1)
    for k in range(words.size):
        counts[words[k]] = counts[words[k]] + V[k]
    return counts

def word_idfs(words, docs):
    counts = np.zeros(np.max(words) + 1)
    for k in range(words.size):
        counts[words[k]] = counts[words[k]] + 1
    return np.log(np.max(docs) + 1) - np.log(counts+1)           # counts has no zero value

def load_uci(basename, min_tf=0, min_tfidf=0):
    vocabfile = os.path.join('uci', 'vocab.%s.txt' % basename)
    docwordfile = os.path.join('uci', 'docword.%s.txt' % basename)
    print "-- Reading files"
    startT = time.time()
    vocab = [w for (w,) in csv.reader(open(vocabfile))]
    with open(docwordfile) as f:
        dn = int(f.readline())
        wn = int(f.readline())
        nnz = int(f.readline())
        idocs = np.zeros(nnz, np.int)
        iwords = np.zeros(nnz, np.int)
        V = np.zeros(nnz, np.float)
        for i in range(nnz):
            entries = map(lambda x: int(x), f.readline().split())
            idocs[i] = entries[0] - 1           # index from 0
            iwords[i] = entries[1] - 1          # index from 0
            V[i] = entries[2]
    print "--- %s seconds ---" % (time.time() - startT)
    if min_tf > 0 or min_tfidf > 0:
        print "-- Trimming vocabulary"
        startT = time.time()
        vocab, Ddw = trim_vacab(vocab, idocs, iwords, V, min_tf=min_tf, min_tfidf=min_tfidf)
        print "--- %s seconds ---" % (time.time() - startT)
    print "-- Constructing co-occurrence matrix"
    startT = time.time()
    Q = compute_Q(Ddw)
    print "--- %s seconds ---" % (time.time() - startT)
    return vocab, Q

def write_topics(fname, vocab, p, r, A, TW):
    with open(fname, 'w') as f:
        nkeyword = np.size(TW, 0)
        ntopic = np.size(TW, 1)
        for t in range(ntopic):
            pt = p[t]
            f.write('%s (%1.3e):\n' % (vocab[pt], r[t]))
            for k in range(nkeyword):
                kw = TW[k, t]
                f.write('\t%-15s (%1.3e)\n' % (vocab[kw], A[kw,t]))
            f.write('\n')
