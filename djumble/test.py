
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print sum([1,8,9,10])*2
print sum([i*2 for i in lst if i in set([1,8,9,10])])
