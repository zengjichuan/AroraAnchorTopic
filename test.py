__author__ = 'admin'

import numpy as np
from scipy import linalg as la
import scipy.sparse as ss
from anchorTopic import *

row = np.array([0,0,1,2,2,2])
col = np.array([0,2,2,0,1,2])
data = np.array([1,2,3,4,5.0,6])
msm = ss.csr_matrix( (data,(row,col)), shape=(3,3))

print msm.todense()
# Q = compute_Q(msm)
p, r = choose_anchor_partial(msm, 2)
#
print (r)