import sys
import random
import numpy as np
import torch
import time
from torchvision.utils import save_image
sys.dont_write_bytecode = True


N=20
# Dimensionality of multiple-choice output
y_dim = N+1
# Sequence length
seq_len = 3*N

cutoff_number=N-3

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
	c = torch.zeros(N+1, 3)
	c[1,0] = 1.
	c[2,2] = 1.
	c[3,:] = 1.

	batch_size = x.shape[0]
	img = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
	img[:,-1] += c[y.long()].view(batch_size, 3, 1, 1)
	img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*(seq_len+1))
	save_image(img, "dist20_hard_test.png")

def get_permutation(batch_size, cutoff):
	permutation_list = []
	for _ in range(batch_size):
		good_permutation = False
		while good_permutation == False:
			perm = np.random.choice(N, size=N, replace=False)
			if set(perm[:cutoff]) != set(np.arange(cutoff)):
				good_permutation = True
		permutation_list.append(perm)
	permutation_list = np.array(permutation_list)
	return permutation_list

def get_test_permutation(batch_size, cutoff):
	margin = N - cutoff
	perm_first = np.stack([np.random.choice(cutoff, size=cutoff, replace=False) for _ in range(batch_size)], axis=0)
	perm_second = np.stack([np.random.choice(margin, size=margin, replace=False) for _ in range(batch_size)],
						   axis=0)
	perm_shift = cutoff * torch.ones_like(torch.Tensor(perm_second))
	perm_second = perm_second + perm_shift.numpy()
	perm_second = perm_second.astype(int)
	perm = np.concatenate([perm_first, perm_second], axis=1)
	return perm

def get_sample(batch_size, prob, shapes, colours):
	# pick 4 shapes
	obj= np.stack([np.random.choice(len(shapes), size=N+1, replace=False) for _ in range(batch_size)],axis=0)
	# take the first 3 out of those chosen shapes for positions 1,2 and 3.
	obj_list=[]
	for i in range(N):
		obj_list.append(obj[:,i])
	x_start =np.stack(obj_list,axis=1)
	# permute those three shapes and remove last one for positions 4 and 5
	perm = get_permutation(batch_size,cutoff_number)
	x_perm = np.stack([x_start[i,perm[i, :]][:-1] for i in range(batch_size)],axis=0)
	x_answers = np.stack([x_start[i,perm[i, :]][-1] for i in range(batch_size)],axis=0)

	# permute the 4 initial objects for the final multiple choice positions 6,7,8 and 9
	perm2 = np.stack([np.random.choice(N+1, size=N+1, replace=False) for _ in range(batch_size)], axis=0)
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

def get_test_sample(batch_size, prob, shapes, colours):
	# pick 4 shapes
	obj= np.stack([np.random.choice(len(shapes), size=N+1, replace=False) for _ in range(batch_size)],axis=0)
	# take the first 3 out of those chosen shapes for positions 1,2 and 3.
	obj_list=[]
	for i in range(N):
		obj_list.append(obj[:,i])
	x_start =np.stack(obj_list,axis=1)
	# permute those three shapes and remove last one for positions 4 and 5
	perm = get_test_permutation(batch_size,cutoff_number)
	x_perm = np.stack([x_start[i,perm[i, :]][:-1] for i in range(batch_size)],axis=0)
	x_answers = np.stack([x_start[i,perm[i, :]][-1] for i in range(batch_size)],axis=0)

	# permute the 4 initial objects for the final multiple choice positions 6,7,8 and 9
	perm2 = np.stack([np.random.choice(N+1, size=N+1, replace=False) for _ in range(batch_size)], axis=0)
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
	#print(all_imgs.shape)
	#get_sample(64, [0.5, 0.5], all_imgs)
	x, y = get_test_sample(32, [0.5, 0.5], all_imgs, all_colours)
	check(x, y)

