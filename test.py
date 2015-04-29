__author__ = 'admin'

import numpy as np
from scipy import linalg as la
import scipy.sparse as ss
from anchorTopic import *
from ioProcess import *
import datetime

starttime = time.time()
row = np.array([0,0,1,2,2,2])
col = np.array([0,2,2,0,1,2])
data = np.array([1,2,3,4,5.0,6])
msm = ss.csr_matrix((data, (row, col)), shape=(3,3))

print msm.todense()
# Q = compute_Q(msm)
# p, r = choose_anchor_partial(msm, 2)
# test nnl
# msm = ss.csr_matrix(msm / msm.sum(1))
# b = msm[2, :]
# AtA = msm * msm.T
# Atb = msm * b.T
# print AtA.todense()
# print Atb.todense()
# x, iter = simplex_nnls_eg(AtA, Atb)
# print (x)
# test compute_A
# Qn = ss.csr_matrix(msm / msm.sum(1))
# s = Qn.sum(1)
# p = np.array([0, 2])
# compute_A(Qn, s, p)

# test io
load_uci('test', min_tf=0.1)

#long running
endtime = time.time()
print endtime - starttime