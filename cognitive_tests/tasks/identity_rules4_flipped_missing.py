import sys
import random
import numpy as np
import torch
import time
from torchvision.utils import save_image
sys.dont_write_bytecode = True

# Dimensionality of multiple-choice output
y_dim = 6
# Sequence length
seq_len = 13

full_list = ["aaaa","abaa" ,"aaba","baaa", "abba", "abab", "aabb", "baca", "bcaa"]
test_codes = ["aaaa","abaa" ,"aaba","baaa", "abba", "abab", "aabb"]
code_list = list(set(full_list)-set(test_codes))

def check(x,y):
	# x-batch_size, 6, 3, 32, 32
	x = torch.Tensor(x)
	y = torch.Tensor(y)

	# Set colors
	c = torch.zeros(6, 3)
	c[1,0] = 1.
	c[2,2] = 1.
	c[3,:] = 1.
	c[4, 1] = 1.
	c[5, :-1] = 1.

	batch_size = x.shape[0]
	img = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
	img[:,-1] += c[y.long()].view(batch_size, 3, 1, 1)
	img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*(seq_len+1))
	save_image(img, "identity_rules4_abca_baca_bcaa.png")

def gen_sample(code, batch_size, obj):
	int_list = []
	shifted_int_list = []
	shift = len(set(list(code)))
	for i in list(code):
		int_list.append(ord(i) - 97)
		shifted_int_list.append(ord(i) - 97 + shift)

	x1 = np.stack([obj[:, int_list[0]], obj[:, int_list[1]], obj[:, int_list[2]], obj[:, int_list[3]]], axis=1)
	x2 = np.stack([obj[:, shifted_int_list[0]], obj[:, shifted_int_list[1]], obj[:, shifted_int_list[2]]], axis=1)
	perm = np.stack([np.random.choice(6, size=6, replace=False) for _ in range(batch_size)], axis=0)
	x_mc = np.stack([obj[i, perm[i, :]] for i in range(batch_size)], axis=0)
	labels = np.stack([np.where(x_mc[i, :] == obj[i, shifted_int_list[3]])[0] for i in range(batch_size)], axis=0)
	x = np.concatenate([x1, x2, x_mc], axis=1)
	y = labels[:, 0]
	return x, y

def get_sample(batch_size, prob, shapes, colours):
	same = int(batch_size*prob[0])
	diff = batch_size-same

	obj = np.stack([np.random.choice(len(shapes), size=6, replace=False) for _ in range(batch_size)], axis=0)
	x_list = []
	y_list = []
	for code in code_list:
		x0, y0 = gen_sample(code, batch_size, obj)
		x_list.append(x0)
		y_list.append(y0)



	x=np.concatenate(x_list, axis=0)
	x = np.reshape(x, [-1])
	x = np.reshape(shapes[x], (len(code_list)*batch_size, -1, 3, 32, 32))

	y = np.concatenate(y_list, axis=0)

	return x, y

def get_test_sample(batch_size, prob, shapes, colours):
	same = int(batch_size*prob[0])
	diff = batch_size-same

	obj = np.stack([np.random.choice(len(shapes), size=6, replace=False) for _ in range(batch_size)], axis=0)
	x_list = []
	y_list = []
	for code in test_codes:
		x0, y0 = gen_sample(code, batch_size, obj)
		x_list.append(x0)
		y_list.append(y0)



	x=np.concatenate(x_list, axis=0)
	x = np.reshape(x, [-1])
	x = np.reshape(shapes[x], (len(test_codes)*batch_size, -1, 3, 32, 32))

	y = np.concatenate(y_list, axis=0)

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
	all_colours = torch.Tensor(all_colours).repeat(1, 1, 32, 32)
	#all_imgs = torch.rand([100, 3, 1, 1]).repeat(1, 1, 32, 32)
	#get_sample(64, [0.5, 0.5], all_imgs)
	x, y = get_sample(12, [0.5, 0.5], all_imgs, all_colours)
	check(x, y)

