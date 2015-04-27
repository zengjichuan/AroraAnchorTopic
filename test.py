__author__ = 'admin'

import numpy as np
from scipy import linalg as la
import scipy.sparse as ss
from anchorTopic import *
import datetime

starttime = time.time()
row = np.array([0,0,1,2,2,2])
col = np.array([0,2,2,0,1,2])
data = np.array([1,2,3,4,5.0,6])
msm = ss.csr_matrix( (data,(row,col)), shape=(3,3))

print msm.todense()
# Q = compute_Q(msm)
# p, r = choose_anchor_partial(msm, 2)
# test nnl
msm = msm / la.norm(msm)
b = msm[2, :]
AtA = msm * msm.T
Atb = msm * b.T
print AtA.todense()
print Atb.todense()
x, iter = simplex_nnls_eg(AtA, Atb)
print (x)


#long running
endtime = time.time()
print endtime - starttime