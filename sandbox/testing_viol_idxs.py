

import numpy as np

must_lnk_con = np.array(
    [[1, 1, 1, 1, 7, 521, 521, 521, 535, 537, 1037, 1057, 1039, 1045, 1098, 1019, 1087],
    [5, 3, 6, 8, 3, 525, 528, 539, 525, 539, 1238,  1358, 1438, 1138, 1038, 1138, 1338]],
    dtype=np.int
)

cannot_lnk_con = np.array(
    [[1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
      7, 7, 538, 548, 558, 738, 938, 838, 555],
     [521,  525,  528,  535,  537,  539,  521,  525,  528,  500,  521,  525,  528,  535,  537,
      539,  521,  535,  537,  539,  521,  525,  528,  535,  537,  539,  521,  525,  528,  535,
      537,  539, 1237, 1357, 1437, 1137, 1037, 1039, 1337]],
    dtype=np.int
)


idxs = np.in1d(must_lnk_con, [1, 5, 528, 521, 1138, 1019, 1087])

idxs = idxs.reshape(2, must_lnk_con.shape[1])

print must_lnk_con[:, np.intersect1d(np.where(idxs[0] != idxs[1])[0], np.hstack((idxs[0].nonzero()[0], idxs[1].nonzero()[0])))]
print

#print np.where((idxs[0] == idxs[1] & idxs[1] == True))

#print idxs
#print must_lnk_con[idxs]
#print np.where((idxs[0] == idxs[1] & idxs[0] == T))

#print cannot_lnk_con[idxs]

#print cannot_lnk_con[np.in1d(cannot_lnk_con[idxs], clstr_idx_arr)]

"""
 ^   ^     ^
 T T T T T T F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F
 T F T F F T T F T F T F T F F T T F F T T F T F F T T F T F F T F F F F F F F

[  1   1   1   1   1   1 521 528 539 521 528 521 528 539 521 539 521 528
 539 521 528 539]

T T T T F T T T F F  F  F  F F  F  T T
T F F F F F T F F F  F  F  F T  F  T F
0       4   6   8 9 10 11 12   14 15
^           ^                      ^
"""
