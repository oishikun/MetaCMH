import cv2
import os
import numpy as np
import h5py


image = []
text = []
tags = []
n = 0
with open(os.path.join('Imagelist', 'Imagelist.txt')) as file:
    lines = file.readlines()
    for line in lines:
        pa = line.split('/')
        path = os.path.join('Flickr', pa[0], pa[1].replace('\n', ''))
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        image.append(img)
        n = n + 1
        print(n, 'images finished\n')
        if n == 100:
            break

images = np.array(image)
images = images.swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)

with open(os.path.join('NUS_WID_Tags', 'AllTags1k.txt')) as file:
    lines = file.readlines()
    for line in lines:
        line = line.split()
        tamp1 = list(map(int, line))
        text.append(tamp1)

text = np.array(text)
text = text[0:100, :]
print(text)
print('text finished\n')

for filename in os.listdir('AllLabels'):
    with open(os.path.join('AllLabels', filename)) as file:
        lines = file.readlines()
        tamp = []
        for line in lines:
            line = line.split()[0]
            tamp.append(int(line))
    tags.append(tamp)
    print('label' + filename + '\n')

tags = np.array(tags)
tags = tags.T
tags = tags[0:100, :]

mat_file = h5py.File('NUS-wide81_mini.mat', 'w')
mat_file.create_dataset('image', data=images)
mat_file.create_dataset('text', data=text)
mat_file.create_dataset('label', data=tags)
mat_file.close()


