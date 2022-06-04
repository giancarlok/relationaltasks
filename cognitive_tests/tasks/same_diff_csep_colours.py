import sys
import random
import numpy as np
import torch
import time
from torchvision.utils import save_image
import torch.nn.functional as F
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
    img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*5)
    #print(img.shape)
    save_image(img, "same_diff_csep_colours.png")

def get_sample(batch_size, prob, shapes, colours):
    h = shapes.shape[-2]
    w = shapes.shape[-1]
    batch_size = 3 * batch_size
    same = int(batch_size*prob[0])
    diff = batch_size-same
    x_shapes = np.random.choice(len(shapes), size=(batch_size, 2), replace=True)
    x_same = np.random.choice(len(colours), size=(same, 1), replace=True)
    x_diff = np.array([np.random.choice(len(colours), size=2, replace=False) for _ in range(diff)])
    x_same = np.concatenate([x_same, x_same], axis=1)
    x_same = np.reshape(x_same, [-1])
    x_same = np.reshape(colours[x_same], (same, -1, 3, h, w))

    x_diff = np.reshape(x_diff, [-1])
    x_diff = np.reshape(colours[x_diff], (diff, -1, 3, h, w))
    x_shapes = np.reshape(x_shapes, [-1])
    x_shapes = np.reshape(shapes[x_shapes], (batch_size, -1, 3, h, w))
    x_c = np.concatenate([x_same, x_diff], axis=0)
    x_shapes = torch.Tensor(x_shapes)
    x_c = torch.Tensor(x_c)

    y = np.zeros(batch_size)
    y[-diff:] = 1
    x1 = x_shapes[:, 0, :, :, :].unsqueeze(1)
    x2 = x_shapes[:, 1, :, :, :].unsqueeze(1)
    xc_1 = x_c[:, 0, :, :, :].unsqueeze(1)
    xc_2 = x_c[:, 1, :, :, :].unsqueeze(1)
    x = torch.cat([x1,xc_1, x2, xc_2], dim=1).numpy()

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
    #all_imgs = F.interpolate(all_imgs, size=(32, 16))
    rng = np.random.RandomState(0)
    all_colours = rng.uniform(size=(100, 3, 1, 1))
    all_colours = torch.Tensor(all_colours).repeat(1, 1, 32, 32)
    #all_colours = F.interpolate(all_colours, size=(32, 16)).numpy()
    x,y = get_sample(64, [0.5,0.5], all_imgs, all_colours)
    check(x,y)

