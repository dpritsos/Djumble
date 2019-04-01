import numpy as np

x = np.array([
    [1, 3, 5, 7, 9], 
    [6, 8, 0, 1, 4],
    [2, 4, 6, 8, 10],
    [1, 0, 3, 2, 2],
    [2, 4, 6, 8, 10],
    [0, 0, 3, 4, 2],
    [2, 6, 6, 8, 10],
    [1, 0, 3, 7, 2],
    [2, 9, 6, 10, 10],
    [0, 3, 3, 0, 4],
], dtype=np.float)


m1 = np.mean(x[0:3], axis=0)
m2 = np.mean(x[4:6], axis=0)
m3 = np.mean(x[7::], axis=0)

dso1 = np.abs(x - m1)
dso2 = np.abs(x - m2)
dso3 = np.abs(x - m3)

m_init1 = np.array([3, 5, 8, 10, 7])
m_init2 = np.array([7, 2, 5, 6, 10])
m_init3 = np.array([1, 10, 4, 7, 5])

ds_init1 = np.abs(x - m_init1)
ds_init2 = np.abs(x - m_init2)
ds_init3 = np.abs(x - m_init3)

dsm1 = np.var(ds_init1[0:3], axis=0)
dsm2 = np.var(ds_init2[4:6], axis=0)
dsm3 = np.var(ds_init3[7::], axis=0)

ds1 = np.abs(ds_init1 - dsm1)
ds2 = np.abs(ds_init2 - dsm2)
ds3 = np.abs(ds_init3 - dsm3)

print dso1, '\n\n', ds1

a = np.array([3,4,5])
b = np.array([8,12])
ma = np.mean(a)
da = a - ma
db = b - ma

print '\n\n' 
print da, db
print np.mean(np.hstack((a,b)))
print np.hstack( (np.sum(da), db) )
print np.sum(np.hstack( (np.sum(da)/2.0 , db) )) / 3.0
