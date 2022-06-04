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
seq_len = 9

def get_sample(batch_size, images, labels):
    bsz = images.shape[0]
    ind = np.random.choice(bsz, size=batch_size, replace=False)
    img = images[ind].squeeze(1)
    img = img.reshape(batch_size, 3, 3, 12, 3, 12).permute(0, 2, 4, 1, 3, 5)
    img = img.reshape(batch_size, 9, 3, 12, 12)
    x = img / 255.
    y = labels[ind].squeeze(1)
    return x, y


#
# def get_sample(batch_size, task_name = "between", set_name = "hexos"):
#     loader = np.load('tasks/mini_{}_{}.npz'.format(task_name, set_name), 'rb')
#     images = loader['images']
#     labels = loader['labels']
#     tasks = loader['tasks']
#     img = torch.Tensor(images).reshape(images.shape[0], 12, 3, 12, 3, 3)
#     img = img.permute(0, 5, 2, 4, 1, 3).reshape(images.shape[0], 9, 3, 12, 12)
#     ind = np.stack([np.random.choice(batch_size, size=1, replace=False) for _ in range(images.shape[0])],axis=0)
#     x = img[ind].squeeze(1)
#     y = labels[ind].squeeze(1)
#     save_image(x, "same_diff_x.png")
#     print("img.max(): ", img.max(), " img.min(): ", img.min())
#
#     return x, y





def check(x,y):
    # x-batch_size, 2, 3, 32, 32
    x= torch.Tensor(x)
    y= torch.Tensor(y)
    batch_size = x.shape[0]
    img = torch.cat([x, torch.zeros_like(x[:, 0: 1])], dim=1)
    img[:,-1] += y.view(batch_size, 1, 1, 1)
    img = img.permute(0,2,3,1,4).reshape(batch_size,3, 12, 12*10)
    #print(img.shape)
    save_image(img, "mini.png")



if __name__=="__main__":
    from PIL import Image

    rng = np.random.RandomState(0)
    x,y = get_sample(32)
    check(x, y)
    # x,y = get_sample(64, [0.5,0.5], all_imgs)
    # check(x,y)

