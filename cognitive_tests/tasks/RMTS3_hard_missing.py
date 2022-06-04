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
# Task segmentation (for context normalization)
task_seg = [[0,1,2], [3,4,5], [6,7,8]]

def check(x,y):
	# x-batch_size, 6, 3, 32, 32
	x= torch.Tensor(x)
	y= torch.Tensor(y)
	batch_size = x.shape[0]
	img = torch.cat([x, torch.zeros_like(x[:, 0: 1])], dim=1)
	img[:,-1] += y.view(batch_size, 1, 1, 1)
	img = img.permute(0,2,3,1,4).reshape(batch_size,3, 32, 32*(seq_len+1))
	save_image(img, "RMTS3_hard_missing.png")

def check_multi(x, y):
	# x-batch_size, 2, 3, 32, 32
	x = torch.Tensor(x)
	y = torch.Tensor(y)
	batch_size = x.shape[0]
	img = torch.cat([x, torch.zeros_like(x[:, 0: 1])], dim=1)
	img[:, -1] += y.view(batch_size, 1, 1, 1)
	img = img.permute(0, 2, 3, 1, 4).reshape(batch_size, 3, 32, 32 * 10)
	# print(img.shape)
	save_image(img, "RMTS_multi.png")

def get_sample(batch_size, prob, shapes, colours):
	same = int(batch_size*prob[0])
	diff = batch_size-same
	obj = np.stack([np.random.choice(len(shapes), size=4, replace=False) for _ in range(batch_size)],axis=0)
	# x_aaa_bcb_1 = np.stack([obj[:,0], obj[:,0], obj[:,0], obj[:,2], obj[:,3], obj[:,2], obj[:,1], obj[:,1],obj[:,1]], axis=1)
	# x_aaa_bcb_2 = np.stack([obj[:,0], obj[:,0], obj[:,0], obj[:,1], obj[:,1],obj[:,1], obj[:,2], obj[:,3], obj[:,2]],axis=1)
	# bern_aaa_bcb = np.random.choice(2, size=(batch_size,1))
	# x_aaa_bcb = x_aaa_bcb_1 * bern_aaa_bcb + x_aaa_bcb_2 * (1 - bern_aaa_bcb)

	x_aaa_cbb_1 = np.stack(
		[obj[:, 0], obj[:, 0], obj[:, 0], obj[:, 3], obj[:, 2], obj[:, 2], obj[:, 1], obj[:, 1],
		 obj[:, 1]], axis=1)

	x_aaa_cbb_2 = np.stack(
		[obj[:, 0], obj[:, 0], obj[:, 0], obj[:, 1], obj[:, 1], obj[:, 1], obj[:, 3],obj[:, 2],
		 obj[:, 2]], axis=1)
	bern_aaa_cbb = np.random.choice(2, size=(batch_size, 1))
	x_aaa_cbb = x_aaa_cbb_1 * bern_aaa_cbb + x_aaa_cbb_2 * (1 - bern_aaa_cbb)
	#x_same = shapes[x_same]



	obj = np.stack([np.random.choice(len(shapes), size=5, replace=False) for _ in range(batch_size)],axis=0)
	# x_aba5_1 = np.stack([obj[:, 0], obj[:, 1], obj[:, 0], obj[:, 4], obj[:, 4], obj[:, 4], obj[:, 2], obj[:, 3], obj[:, 2]], axis=1)
	# x_aba5_2 = np.stack([obj[:, 0], obj[:, 1], obj[:, 0], obj[:, 2], obj[:, 3], obj[:, 2], obj[:, 4], obj[:, 4], obj[:, 4]], axis=1)
	# bern_aba5 = np.random.choice(2, size=(batch_size, 1))
	# x_aba5 = x_aba5_1 * bern_aba5 + x_aba5_2 * (1 - bern_aba5)

	x_abb5_1 = np.stack(
		[obj[:, 0], obj[:, 1], obj[:, 1], obj[:, 4], obj[:, 4], obj[:, 4], obj[:, 2], obj[:, 3], obj[:, 3]], axis=1)
	x_abb5_2 = np.stack(
		[obj[:, 0], obj[:, 1], obj[:, 1], obj[:, 2], obj[:, 3], obj[:, 3], obj[:, 4], obj[:, 4], obj[:, 4]], axis=1)
	bern_abb5 = np.random.choice(2, size=(batch_size, 1))
	x_abb5 = x_abb5_1 * bern_abb5 + x_abb5_2 * (1 - bern_abb5)
	#x_diff = shapes[x_diff]

	obj = np.stack([np.random.choice(len(shapes), size=6, replace=False) for _ in range(batch_size)], axis=0)
	# x_aba6_1 = np.stack(
	# 	[obj[:, 0], obj[:, 1], obj[:, 0], obj[:, 4], obj[:, 5], obj[:, 5], obj[:, 2], obj[:, 3], obj[:, 2]], axis=1)
	# x_aba6_2 = np.stack(
	# 	[obj[:, 0], obj[:, 1], obj[:, 0], obj[:, 2], obj[:, 3], obj[:, 2], obj[:, 4], obj[:, 5], obj[:, 5]], axis=1)
	# bern_aba6 = np.random.choice(2, size=(batch_size, 1))
	# x_aba6 = x_aba6_1 * bern_aba6 + x_aba6_2 * (1 - bern_aba6)

	x_abb6_1 = np.stack(
		[obj[:, 0], obj[:, 1], obj[:, 1], obj[:, 4], obj[:, 5], obj[:, 4], obj[:, 2], obj[:, 3], obj[:, 3]], axis=1)
	x_abb6_2 = np.stack(
		[obj[:, 0], obj[:, 1], obj[:, 1], obj[:, 2], obj[:, 3], obj[:, 3], obj[:, 4], obj[:, 5], obj[:, 4]], axis=1)
	bern_abb6 = np.random.choice(2, size=(batch_size, 1))
	x_abb6 = x_abb6_1 * bern_abb6 + x_abb6_2 * (1 - bern_abb6)


	x=np.concatenate([x_aaa_cbb, x_abb5, x_abb6], axis=0)
	x = np.reshape(x, [-1])

	x = np.reshape(shapes[x], (3*batch_size, -1, 3, 32, 32))

	y=np.concatenate([bern_aaa_cbb, bern_abb5, bern_abb6], axis=0)[:,0]

	return x, y

def get_test_sample(batch_size, prob, shapes, colours):
	same = int(batch_size*prob[0])
	diff = batch_size-same
	obj = np.stack([np.random.choice(len(shapes), size=7, replace=False) for _ in range(batch_size)],axis=0)
	x_aab_cde_1 = np.stack([obj[:,0], obj[:,0], obj[:,1], obj[:,4], obj[:,5], obj[:,6], obj[:,2], obj[:,2], obj[:,3]], axis=1)
	x_aab_cde_2 = np.stack([obj[:,0], obj[:,0], obj[:,1], obj[:,2], obj[:,2], obj[:,3], obj[:,4], obj[:,5], obj[:,6]],axis=1)
	bern_aab_cde = np.random.choice(2, size=(batch_size,1))
	x_aab_cde = x_aab_cde_1 * bern_aab_cde + x_aab_cde_2 * (1 - bern_aab_cde)

	obj = np.stack([np.random.choice(len(shapes), size=8, replace=False) for _ in range(batch_size)], axis=0)
	x_abc_dde_1 = np.stack(
		[obj[:, 0], obj[:, 1], obj[:, 2], obj[:, 3], obj[:, 3], obj[:, 4], obj[:, 5], obj[:, 6], obj[:, 7]], axis=1)
	x_abc_dde_2 = np.stack(
		[obj[:, 0], obj[:, 1], obj[:, 2], obj[:, 5], obj[:, 6], obj[:, 7], obj[:, 3], obj[:, 3], obj[:, 4]], axis=1)
	bern_abc_dde = np.random.choice(2, size=(batch_size, 1))
	x_abc_dde = x_abc_dde_1 * bern_abc_dde + x_abc_dde_2 * (1 - bern_abc_dde)


	x=np.concatenate([x_aab_cde, x_abc_dde], axis=0)
	x = np.reshape(x, [-1])

	x = np.reshape(shapes[x], (2*batch_size, -1, 3, 32, 32))

	y=np.concatenate([bern_aab_cde, bern_abc_dde], axis=0)[:,0]

	return x, y


def get_coloured_sample(batch_size, prob, shapes, colours):
	batch_size = 3 * batch_size
	same = int(batch_size*prob[0])
	diff = batch_size-same
	obj = np.stack([np.random.choice(len(colours), size=4, replace=False) for _ in range(same)],axis=0)
	obj_1 = np.stack([obj[:,0], obj[:,0], obj[:,1], x_same[:,1], x_same[:,2], x_same[:,3]],axis=1)
	x_same_2 = np.stack([x_same[:, 0], x_same[:, 0], x_same[:, 2], x_same[:, 3], x_same[:, 1], x_same[:, 1]], axis=1)
	bern_same = np.random.choice(2, size=(same,1))
	x_same = x_same_1 * bern_same + x_same_2 * (1 - bern_same)
	#x_same = shapes[x_same]


	x_diff = np.stack([np.random.choice(len(colours), size=5, replace=False) for _ in range(diff)],axis=0)
	x_diff_1 = np.stack([x_diff[:, 0], x_diff[:, 1], x_diff[:, 2], x_diff[:, 2], x_diff[:, 3], x_diff[:, 4]], axis=1)
	x_diff_2 = np.stack([x_diff[:, 0], x_diff[:, 1], x_diff[:, 3], x_diff[:, 4], x_diff[:, 2], x_diff[:, 2]], axis=1)
	bern_diff = np.random.choice(2, size=(diff, 1))
	x_diff = x_diff_1 * bern_diff + x_diff_2 * (1 - bern_diff)
	#x_diff = shapes[x_diff]


	x=np.concatenate([x_same, x_diff], axis=0)
	x = np.reshape(x, [-1])
	x = np.reshape(colours[x], (batch_size, -1, 3, 32, 32))

	y=np.concatenate([1-bern_same, bern_diff], axis=0)[:,0]

	#print(x.shape, y.shape)

	return x, y

def get_multi_sample(batch_size, prob, shapes):
	batch_size = 3 * batch_size
	same = int(batch_size*prob[0])
	diff = batch_size-same
	x_same = np.stack([np.random.choice(len(shapes), size=4, replace=False) for _ in range(same)],axis=0)
	x_same_1 = np.stack([x_same[:,0], x_same[:,0], x_same[:,1], x_same[:,1], x_same[:,2], x_same[:,3]],axis=1)
	x_same_2 = np.stack([x_same[:, 0], x_same[:, 0], x_same[:, 2], x_same[:, 3], x_same[:, 1], x_same[:, 1]], axis=1)
	bern_same = np.random.choice(2, size=(same,1))
	x_same = x_same_1 * bern_same + x_same_2 * (1 - bern_same)
	#x_same = shapes[x_same]


	x_diff = np.stack([np.random.choice(len(shapes), size=5, replace=False) for _ in range(diff)],axis=0)
	x_diff_1 = np.stack([x_diff[:, 0], x_diff[:, 1], x_diff[:, 2], x_diff[:, 2], x_diff[:, 3], x_diff[:, 4]], axis=1)
	x_diff_2 = np.stack([x_diff[:, 0], x_diff[:, 1], x_diff[:, 3], x_diff[:, 4], x_diff[:, 2], x_diff[:, 2]], axis=1)
	bern_diff = np.random.choice(2, size=(diff, 1))
	x_diff = x_diff_1 * bern_diff + x_diff_2 * (1 - bern_diff)
	#x_diff = shapes[x_diff]


	x=np.concatenate([x_same, x_diff], axis=0)
	x = np.reshape(x, [-1])
	x = np.reshape(shapes[x], (batch_size, -1, 3, 32, 32))
	x_zeros = np.zeros([batch_size, 3, 3, 32, 32])
	x = np.concatenate([x, x_zeros], axis=1)

	y=np.concatenate([1-bern_same, bern_diff], axis=0)[:,0]

	#print(x.shape, y.shape)

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
	x,y = get_test_sample(64, [0.5,0.5], all_imgs, all_colours)
	check_multi(x, y)

	# x, y = get_sample(64, [0.5, 0.5], all_imgs)
	# check(x, y)

