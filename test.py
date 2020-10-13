import tqdm
import numpy as np
import torch as t
import time
import os


# a = np.random.randint(1, 4, (100, 10))
# label = (a == 1)
# print(a)
# print(label)
# index = np.arange(100)
#
# ind = index[label[:, 2] == 1]
# print(ind)

# dir_name = 'hashcodes' + time.strftime("%Y_%m_%d_%H%M", time.localtime())
# os.makedirs(dir_name)
# mlist_i2t = []
# mlist_t2i = []
# for i in range(10):
#     map1 = np.random.randint(1, 10)
#     map2 = np.random.randint(1, 10)
#     mlist_t2i.append(map1)
#     mlist_i2t.append(map2)
#
# f = open(os.path.join(dir_name, 'maplist.txt'), "w")
# f.write("i2t: " + str(mlist_i2t))
# f.write("\n")
# f.write("t2i: " + str(mlist_t2i))


path = './data/text_model.pth'
a = t.load(path, map_location=t.device('cpu'))
print(a)