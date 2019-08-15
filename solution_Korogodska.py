# -*- coding: utf-8 -*-

import argparse
import glob
import sys
import os.path
from time import time
from PIL import Image
import numpy as np


def img_similarity(image1, image2):
    s = 0
    for band_index, band in enumerate(image1.getbands()):
        m1 = np.array([p[band_index] for p in image1.getdata()]).reshape(*image1.size)
        m2 = np.array([p[band_index] for p in image2.getdata()]).reshape(*image2.size)
        s += np.sum(np.abs(m1-m2))
    return s


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="First test task on images similarity.")
    parser.add_argument('--path', help='folder with images', required=True)
    args=parser.parse_args()
    t = time()

path = args.path
block_size = 20
threshold = 0.1
full_diff = (block_size**2)*3*255

if not os.path.exists(path):
    print("Folder doesn't exist")
    sys.exit()


images = [[], []]
result = []

for f in glob.glob("".join([path,"/*"])):
    images[0].append(Image.open(f).resize((block_size, block_size), Image.BILINEAR))
    images[1].append(f.replace(path, ""))
    
for i in range(len(images[0])):
    for j in range(len(images[0])):
        if (i == j):
            result.append([-1, images[1][i], images[1][j]])
        else:
            result.append([img_similarity(images[0][i], images[0][j]) / full_diff, images[1][i], images[1][j]])
            
result_fin = np.array(list(filter(lambda x: (0 <= x[0] <= threshold), result)))[:, 1:]
result_fin.sort()
result_fin = [list(x) for x in set(tuple(x) for x in result_fin)]

print("\n".join([str(' '.join(str(j) for j in i)) for i in result_fin]))
print("Time running:", time() - t, "sec.")
