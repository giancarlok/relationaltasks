from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch


def access_coord(img,r,c):
    return img[12*r:12*(r+1),12*c:12*(c+1),:]

def nonzero_coord(img):
    nonzero_list =[]
    for c in range(3):
        for r in range(3):
            if not np.array_equal(access_coord(img ,r ,c) ,np.zeros((12 ,12 ,3) ,dtype=int)):
                nonzero_list.append((r ,c))
    return nonzero_list

def return_colour(x):
    colour_found=False
    r=c=0
    colour = np.array([0, 0, 0])
    while r<12 and c<12 and not colour_found:
        if np.array_equal(x[r,c],[0 , 0 , 0]):
            if c<11:
                c+=1
            elif r<11:
                c=0
                r+=1
        else:
            colour_found=True
            colour = x[r,c]

    return colour

def luminance(x):
    R,G,B = return_colour(x)
    return 0.2126*R + 0.7152*G + 0.0722*B

def diff_luminance(x):
    coord = nonzero_coord(x)
    r0, c0= coord[0]
    r1, c1= coord[1]
    l0 = luminance(access_coord(x,r0,c0))
    l1 = luminance(access_coord(x,r1,c1))
    return l0 != l1

def antisymmetric_leftof(x):
    coord = nonzero_coord(x)
    r0, c0= coord[0]
    r1, c1= coord[1]
    l0 = luminance(access_coord(x,r0,c0))
    l1 = luminance(access_coord(x,r1,c1))
    if (l1<l0 and c1<c0 and r1<=r0) or (l1<l0 and c1<=c0 and r0<r1) or (l0<l1 and c0<c1 and r0 <=r1) or (l0 <l1 and c0 <=c1 and r1< r0):
        return True
    else:
        return False

def create_leftof_dataset(object_set):
    loader = np.load('../npz_files/{}_{}.npz'.format("same", object_set), 'rb')
    images = loader['images']
    tasks = loader['tasks']
    flipped_images = np.array([np.flip(np.flip(img,axis=0),axis=1) for img in images])
    augmented_images = np.concatenate((images, flipped_images),axis=0)
    luminance_labels = np.array([diff_luminance(augmented_images[i]) for i in range(len(augmented_images))])
    images = augmented_images[luminance_labels]
    label_list=[]
    for img in images:
        label_list.append([antisymmetric_leftof(img)])

    labels = np.array(label_list)
    print(object_set, labels.sum(axis=0)/len(labels))
    with open('../npz_files/{}_{}.npz'.format("leftof_dataset", object_set), 'wb') as outfile:
        np.savez_compressed(outfile, images=images, labels=labels, tasks=tasks)

if __name__=="__main__":
    create_leftof_dataset("pentos")
    create_leftof_dataset("hexos")
    create_leftof_dataset("stripes")