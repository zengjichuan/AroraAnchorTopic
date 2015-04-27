# Introduction
# ============
#
# This code computes a word-topic matrix using the spectral approach
# described in an [ICML 2013][1] paper.  The main algorithm has two
#( phases:
#
# 1. In the first phase, we compute "anchor words" associated
#    with specific topics.
#
# 2. In the second phase, we compute the word-topic intensities
#    by a sequence of constrained QPs
#
# [1]: http://mimno.infosci.cornell.edu/papers/arora13.pdf
#

__author__ = 'admin'

import numpy as np
from scipy import linalg as la
import scipy.sparse as ss
import time

def choose_anchor_partial(A, k):
    # copy matrix A
    B = A.copy()
    B.data **= 2
    cnorm2 = B.sum(1)

    p = np.zeros(k, np.int)
    Q = np.zeros((k, np.size(A, 1)))
    r = np.zeros(k)
    for j in range(k):
        p[j] = np.argmax(cnorm2)
        # tmp1 = A.getrow(p[j])
        # tmp2 = np.matrix(Q[0:j, :]).T
        # tmp3 = tmp1 * tmp2
        # tmp4 = tmp3 * np.matrix(Q[0:j, :])
        # Q[j, :] = tmp1 - tmp4
        Q[j, :] = A.getrow(p[j]) - (A.getrow(p[j]) * np.matrix(Q[0:j, :]).T) * np.matrix(Q[0:j, :])
        r[j] = la.norm(Q[j, :])
        Q[j, :] = Q[j, :] / r[j]
        # tmp5 = np.matrix(Q[j, :]).T
        # tmp6 = A * tmp5
        # tmp7 = np.square(tmp6)
        # cnorm2 = cnorm2 - tmp7
        cnorm2 = cnorm2 - np.square(A * np.matrix(Q[j, :]).T)
    return p, r

# X_(i,j) denotes doc i in word j
def compute_Q(X):
    print 'Forming Q'
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

def simplex_nnls_eg(AtA, Atb, maxIter = 500):
    etat = 10
    epsilon = 1e-5
    # initialization
    K = np.size(Atb, 0)
    x = np.ones((K, 1)) / K
    converge = False
    iterNum = 0
    while not converge:
        p = 2 * AtA * x - 2 * Atb
        tmp1 = np.exp(-etat * p)
        x = np.multiply(x, tmp1)
        x = x / la.norm(x)
        pprime = 2 * AtA * x - 2 * Atb
        lmd = pprime - np.ones((K, 1)) * np.min(pprime)
        if lmd.T * x < epsilon:
            converge = True
        iterNum += 1
        if iterNum == maxIter:
            converge = True
        etat *= 0.99
    return x, iterNum

def compute_A(Qn, s, p):
    Tt = Qn[p, :]
    AtA = Tt * Tt.T
    AtB = Tt * Qn.T
    K = np.size(Tt, 0)
    V = np.size(Tt, 1)
    C = np.zeros((V, K))
    maxerr1 = 0
    maxerr2 = 0
    titer = 0
    for i in range(V):
        Atb = AtB[:, i]
        x, iter = simplex_nnls_eg(AtA, Atb)
        C[i, :] = x.T * s[i].item(0)
        # check normalization error
        maxerr1 = max(maxerr1, abs(sum(x) - 1))
        r = AtA * x - Atb
        phi = 2 * (r * min(r)).T * x
        maxerr2 = max(maxerr2, phi)
        titer += iter
    print "Max error:", maxerr1, maxerr2
    print "Total iteration: ", titer
    return C / C.sum(0)

def mine_topics(Q, ntopic = 100, nword = 20):
    Qn = Q / Q.sum(1)
    s = Q.sum(1)
    print("-- Timing anchor words with partial fact")
    startT = time.time()
    p, r = choose_anchor_partial(Qn, ntopic)
    print "--- %s seconds ---" % (time.time() - startT)
    startT = time.time()
    print "-- Compute intensities"
    A = compute_A(Qn, s, p)
    print "--- %s seconds ---" % (time.time() - startT)
    startT = time.time()
    print "-- Find top words per topic"
    TW = np.argsort(-A, axis=0)
    print "--- %s seconds ---" % (time.time() - startT)
    return p, r, A, TW


