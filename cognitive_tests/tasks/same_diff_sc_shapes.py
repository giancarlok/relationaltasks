import sys
import random
import numpy as np
import torch
import time
from torchvision.utils import save_image
sys.dont_write_bytecode = True

# Dimensionality of multiple-choice output
y_dim = 2
# Sequence length
seq_len = 2

def check(x,y):
    # x-batch_size, 2, 3, 32, 32
    x= torch.Tensor(x)
    y= torch.Tensor(y)
    batch_size = x.shape[0]
    img = torch.cat([x, torch.zeros_like(x[:, 0: 1])], dim=1)
    img[:,-1] += y.view(batch_size, 1, 1, 1)
    img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*3)
    #print(img.shape)
    save_image(img, "same_diff_sc_shapes.png")

def check_multi(x,y):
    # x-batch_size, 2, 3, 32, 32
    x= torch.Tensor(x)
    y= torch.Tensor(y)
    batch_size = x.shape[0]
    img = torch.cat([x, torch.zeros_like(x[:, 0: 1])], dim=1)
    img[:,-1] += y.view(batch_size, 1, 1, 1)
    img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*10)
    #print(img.shape)
    save_image(img, "same_diff_multi.png")
# def get_sample(batch_size, prob, shapes):
#     same = int(batch_size*prob[0])
#     diff = batch_size-same
#     x_same = np.random.choice(len(shapes), same)
#     x_diff = [np.random.choice(len(shapes), size=2, replace=False) for _ in range(diff)]
#     x_same = np.expand_dims(shapes[x_same], axis=1)
#     x_same = np.concatenate([x_same,x_same], axis=1)
#     x_diff = shapes[x_diff]
#     y = np.zeros(batch_size)
#     y[-diff:] = 1
#     x = np.concatenate([x_same, x_diff], axis=0)
#
#     return x, y

def get_sample(batch_size, prob, shapes, colours):
    batch_size = 3 * batch_size
    same = int(batch_size*prob[0])
    diff = batch_size-same
    x_col = np.array([np.random.choice(len(colours), size=2, replace=True) for _ in range(batch_size)])
    x_same = np.stack([np.random.choice(len(shapes), size=1, replace=False) for _ in range(same)],axis=0)
    x_diff = np.array([np.random.choice(len(shapes), size=2, replace=False) for _ in range(diff)])
    x_same = np.concatenate([x_same, x_same], axis=1)
    x_same = np.reshape(x_same, [-1])
    x_same = np.reshape(shapes[x_same], (same, -1, 3, 32, 32))

    x_diff = np.reshape(x_diff, [-1])
    x_diff = np.reshape(shapes[x_diff], (diff, -1, 3, 32, 32))
    x_c = np.reshape(x_col, [-1])
    x_c = np.reshape(colours[x_c], (batch_size, -1, 3, 32, 32))


    # x_same = np.expand_dims(x_same, axis=1)
    # print("x_same 4", x_same.shape)
    y = np.zeros(batch_size)
    y[-diff:] = 1
    x = np.concatenate([x_same, x_diff], axis=0)*x_c


    return x, y

def get_tricky_sample(batch_size, prob, shapes, colours, weight):
    batch_size = 3 * batch_size
    same = int(batch_size*prob[0])
    diff = batch_size-same
    w_same = int(weight*same)
    w_diff = int(weight * diff)
    x_col_diff_same = np.array([np.random.choice(len(colours), size=2, replace=False) for _ in range(w_same)])
    x_col_same_same = np.stack([np.random.choice(len(colours), size=1, replace=False) for _ in range(same-w_same)], axis=0)
    x_col_same_same = np.concatenate([x_col_same_same, x_col_same_same], axis=1)

    x_col_diff_diff = np.array([np.random.choice(len(colours), size=2, replace=False) for _ in range(diff-w_diff)])
    x_col_same_diff = np.stack([np.random.choice(len(colours), size=1, replace=False) for _ in range(w_diff)], axis=0)
    x_col_same_diff = np.concatenate([x_col_same_diff, x_col_same_diff], axis=1)
    x_col = np.concatenate([x_col_diff_same, x_col_same_same, x_col_same_diff, x_col_diff_diff], axis=0)

    x_c = np.reshape(x_col, [-1])
    x_c = np.reshape(colours[x_c], (batch_size, -1, 3, 32, 32))

    x_same = np.stack([np.random.choice(len(shapes), size=1, replace=False) for _ in range(same)],axis=0)
    x_diff = np.array([np.random.choice(len(shapes), size=2, replace=False) for _ in range(diff)])

    x_same = np.concatenate([x_same, x_same], axis=1)
    x_same = np.reshape(x_same, [-1])
    x_same = np.reshape(shapes[x_same], (same, -1, 3, 32, 32))

    x_diff = np.reshape(x_diff, [-1])
    x_diff = np.reshape(shapes[x_diff], (diff, -1, 3, 32, 32))


    # x_same = np.expand_dims(x_same, axis=1)
    # print("x_same 4", x_same.shape)
    y = np.zeros(batch_size)
    y[-diff:] = 1
    x = np.concatenate([x_same, x_diff], axis=0)*x_c

    return x, y

def get_shapes_sample(batch_size, prob, shapes, colours):
    batch_size = 3 * batch_size
    same = int(batch_size*prob[0])
    diff = batch_size-same
    x_same = np.stack([np.random.choice(len(shapes), size=1, replace=False) for _ in range(same)],axis=0)
    x_diff = np.array([np.random.choice(len(shapes), size=2, replace=False) for _ in range(diff)])
    x_same = np.concatenate([x_same, x_same], axis=1)
    x_same = np.reshape(x_same, [-1])
    x_same = np.reshape(shapes[x_same], (same, -1, 3, 32, 32))

    x_diff = np.reshape(x_diff, [-1])
    x_diff = np.reshape(shapes[x_diff], (diff, -1, 3, 32, 32))


    # x_same = np.expand_dims(x_same, axis=1)
    # print("x_same 4", x_same.shape)
    y = np.zeros(batch_size)
    y[-diff:] = 1
    x = np.concatenate([x_same, x_diff], axis=0)

    return x, y

def get_coloured_sample(batch_size, prob, shapes, colours):
    batch_size = 3 * batch_size
    same = int(batch_size*prob[0])
    diff = batch_size-same
    x_same = np.stack([np.random.choice(len(colours), size=1, replace=False) for _ in range(same)],axis=0)
    x_diff = np.array([np.random.choice(len(colours), size=2, replace=False) for _ in range(diff)])
    x_same = np.concatenate([x_same, x_same], axis=1)
    x_same = np.reshape(x_same, [-1])
    x_same = np.reshape(colours[x_same], (same, -1, 3, 32, 32))

    x_diff = np.reshape(x_diff, [-1])
    x_diff = np.reshape(colours[x_diff], (diff, -1, 3, 32, 32))


    # x_same = np.expand_dims(x_same, axis=1)
    # print("x_same 4", x_same.shape)
    y = np.zeros(batch_size)
    y[-diff:] = 1
    x = np.concatenate([x_same, x_diff], axis=0)

    return x, y

if __name__=="__main__":
    from PIL import Image

    all_imgs = []
    for i in range(100):
        img_fname = './imgs/' + str(i) + '.png'
        img = torch.Tensor(np.array(Image.open(img_fname))) / 255.
        img=img.repeat(1,3,1,1)
        all_imgs.append(img)
    all_imgs = torch.stack(all_imgs, 0)[:,0]
    rng = np.random.RandomState(0)
    all_colours = rng.uniform(size=(100, 3, 1, 1))
    all_colours = torch.Tensor(all_colours).repeat(1, 1, 32, 32).numpy()
    x,y = get_sample(64, [0.5, 0.5], all_imgs, all_colours)
    #x, y = get_tricky_sample(64, [0.5, 0.5], all_imgs, all_colours, 0.9)
    # x,y = get_multi_sample(64, [0.5,0.5], all_imgs)
    # check_multi(x, y)
    # x,y = get_sample(64, [0.5,0.5], all_imgs)
    check(x,y)

