import numpy as np
b = np.fromfile("../labels/FCVID/test_y.binary", dtype='int64')
f = open("./result/test_result.txt").readlines()
logits = list(map(lambda x:int(x.strip().split(' ')[1]), f))
print(np.mean(logits == b))
