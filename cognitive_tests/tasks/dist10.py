import sys
import random
import numpy as np
import torch
import time
from torchvision.utils import save_image
sys.dont_write_bytecode = True

# Dimensionality of multiple-choice output
y_dim = 11
# Sequence length
seq_len = 30

# def check(x,y):
# 	# x-batch_size, 6, 3, 32, 32
# 	x = torch.Tensor(x)
# 	y = torch.Tensor(y)
# 	batch_size = x.shape[0]
# 	img = torch.cat([x, torch.zeros_like(x[:, 0: 1])], dim=1)
# 	img[:,-1] += y.view(batch_size, 1, 1, 1)
# 	img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*10)
# 	save_image(img, "dist3.png")

def check(x,y):
	# x-batch_size, 6, 3, 32, 32
	x = torch.Tensor(x)
	y = torch.Tensor(y)

	# Set colors
	c = torch.zeros(4, 3)
	c[1,0] = 1.
	c[2,2] = 1.
	c[3,:] = 1.

	batch_size = x.shape[0]
	img = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
	img[:,-1] += c[y.long()].view(batch_size, 3, 1, 1)
	img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*10)
	save_image(img, "dist10.png")

def get_sample(batch_size, prob, shapes, colours):
	batch_size = 3*batch_size
	same = int(batch_size*prob[0])
	diff = batch_size-same
	# pick 4 shapes
	obj= np.stack([np.random.choice(len(shapes), size=11, replace=False) for _ in range(batch_size)],axis=0)
	# take the first 3 out of those chosen shapes for positions 1,2 and 3.
	x_start =np.stack([obj[:,0], obj[:,1], obj[:,2], obj[:,3], obj[:,4], obj[:,5], obj[:,6], obj[:,7], obj[:,8], obj[:,9]],axis=1)
	# permute those three shapes and remove last one for positions 4 and 5
	perm = np.stack([np.random.choice(10, size=10, replace=False) for _ in range(batch_size)],axis=0)
	x_perm = np.stack([x_start[i,perm[i, :]][:-1] for i in range(batch_size)],axis=0)
	x_answers = np.stack([x_start[i,perm[i, :]][-1] for i in range(batch_size)],axis=0)

	# permute the 4 initial objects for the final multiple choice positions 6,7,8 and 9
	perm2 = np.stack([np.random.choice(11, size=11, replace=False) for _ in range(batch_size)], axis=0)
	x_mc = np.stack([obj[i,perm2[i, :]] for i in range(batch_size)],axis=0)
	labels = np.stack([np.where(x_mc[i,:] == x_answers[i])[0] for i in range(batch_size)],axis=0)
	x = np.concatenate([x_start,x_perm, x_mc], axis=1)
	# print("x", x.shape)
	# print("shapes", shapes.shape)
	x = np.reshape(x, [-1])
	x = np.reshape(shapes[x], (batch_size, -1, 3, 32, 32))

	#print("x after", x.shape)
	y = labels[:,0]



	return x, y


def get_coloured_sample(batch_size, prob, shapes, colours):
	batch_size = 3*batch_size
	same = int(batch_size*prob[0])
	diff = batch_size-same
	# pick 4 shapes
	obj= np.stack([np.random.choice(len(colours), size=11, replace=False) for _ in range(batch_size)],axis=0)
	# take the first 3 out of those chosen shapes for positions 1,2 and 3.
	x_start =np.stack([obj[:,0], obj[:,1], obj[:,2], obj[:,3], obj[:,4], obj[:,5], obj[:,6], obj[:,7], obj[:,8], obj[:,9]],axis=1)
	# permute those three shapes and remove last one for positions 4 and 5
	perm = np.stack([np.random.choice(10, size=10, replace=False) for _ in range(batch_size)],axis=0)
	x_perm = np.stack([x_start[i,perm[i, :]][:-1] for i in range(batch_size)],axis=0)
	x_answers = np.stack([x_start[i,perm[i, :]][-1] for i in range(batch_size)],axis=0)

	# permute the 4 initial objects for the final multiple choice positions 6,7,8 and 9
	perm2 = np.stack([np.random.choice(11, size=11, replace=False) for _ in range(batch_size)], axis=0)
	x_mc = np.stack([obj[i,perm2[i, :]] for i in range(batch_size)],axis=0)
	labels = np.stack([np.where(x_mc[i,:] == x_answers[i])[0] for i in range(batch_size)],axis=0)
	x = np.concatenate([x_start,x_perm, x_mc], axis=1)
	# print("x", x.shape)
	# print("shapes", shapes.shape)
	x = np.reshape(x, [-1])
	x = np.reshape(colours[x], (batch_size, -1, 3, 32, 32))

	#print("x after", x.shape)
	y = labels[:,0]



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
	#print(all_imgs.shape)
	#get_sample(64, [0.5, 0.5], all_imgs)
	x, y = get_sample(32, [0.5, 0.5], all_imgs)
	check(x, y)

