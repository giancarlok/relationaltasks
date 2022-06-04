import torch
import torch.nn as nn
import torch.nn.functional as F
from util import log

class Encoder_conv(nn.Module):
	def __init__(self, args):
		super(Encoder_conv, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# Fully-connected layers
		log.info('FC layers...')
		self.fc1 = nn.Linear(32, 256)
		self.fc2 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		#print("x shape", x.shape)
		x=x.squeeze(1)
		#print(self.conv1.weight.grad)
		conv1_out = self.relu(self.conv1(x))
		#print("conv1_out", conv1_out.shape)
		conv2_out = self.relu(self.conv2(conv1_out))
		#print("conv2_out", conv2_out.shape)
		conv3_out = self.relu(self.conv3(conv2_out))
		#print("conv3_out", conv3_out.shape)
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		#print("conv3_out_flat", conv3_out_flat.shape)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(conv3_out_flat))
		#print("fc1_out", fc1_out.shape)
		fc2_out = self.relu(self.fc2(fc1_out))
		#print("fc2_out", fc2_out.shape)
		# Output
		z = fc2_out
		return z

class Predinet_Encoder(nn.Module):
	def __init__(self, args):
		super(Predinet_Encoder, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = nn.Conv2d(3, 32, 12, stride=6, padding=0)
		# self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# # Fully-connected layers
		# log.info('FC layers...')
		# self.fc1 = nn.Linear(32, 256)
		# self.fc2 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		#print("x shape", x.shape)
		x=x.squeeze(1)
		#print(self.conv1.weight.grad)
		z = self.relu(self.conv1(x))
		print("z out", z.shape)
		#print("conv1_out", conv1_out.shape)
		#conv2_out = self.relu(self.conv2(conv1_out))
		#print("conv2_out", conv2_out.shape)
		#conv3_out = self.relu(self.conv3(conv2_out))
		#print("conv3_out", conv3_out.shape)
		# Flatten output of conv. net
		z_flat = torch.flatten(z, 1)
		print("z flat out", z_flat.shape)
		#print("conv3_out_flat", conv3_out_flat.shape)
		# Fully-connected layers
		#fc1_out = self.relu(self.fc1(conv3_out_flat))
		#print("fc1_out", fc1_out.shape)
		#fc2_out = self.relu(self.fc2(fc1_out))
		#print("fc2_out", fc2_out.shape)
		# Output
		#z = fc2_out
		return z_flat

class Encoder_conv_6channels(nn.Module):
	def __init__(self, args):
		super(Encoder_conv_6channels, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = nn.Conv2d(6, 32, 4, stride=2, padding=1)
		# for param in self.conv1.parameters():
		# 	print("freezing params")
		# 	param.requires_grad = False
		#copy_shape = self.conv1.weight[:, 3:, :, :]
		#self.conv1.weight[:, 3:, :, :] = 0


		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# Fully-connected layers
		log.info('FC layers...')
		self.fc1 = nn.Linear(4*4*32, 256)
		self.fc2 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')

		#self.conv1.weight.data[:, 3:, :, :] = 0

	def forward(self, x):
		# Convolutional layers
		#print("x shape", x.shape)
		#print(self.conv1.weight[:, 3:, :, :])
		# print(self.conv1.weight.shape)
		conv1_out = self.relu(self.conv1(x))
		conv2_out = self.relu(self.conv2(conv1_out))
		conv3_out = self.relu(self.conv3(conv2_out))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(conv3_out_flat))
		fc2_out = self.relu(self.fc2(fc1_out))
		# Output
		z = fc2_out
		return z

class Encoder_conv_l1(nn.Module):
	def __init__(self, args):
		super(Encoder_conv_l1, self).__init__()
		self.z_size = 4
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# Fully-connected layers
		log.info('FC layers...')
		self.fc1 = nn.Linear(4*4*32, 256)
		self.fc2 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		self.w1 = nn.Linear(128, self.z_size)
		self.w2 = nn.Linear(128, self.z_size)
		self.select = nn.Linear(128, 8)
		self.signature = nn.Parameter(torch.randn(2, 8))
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		#print("x shape", x.shape)
		conv1_out = self.relu(self.conv1(x))
		conv2_out = self.relu(self.conv2(conv1_out))
		conv3_out = self.relu(self.conv3(conv2_out))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(conv3_out_flat))
		#l1_penalty_fc1 = sum(p.abs().sum()
						 #for p in self.fc1.parameters())

		fc2_out = self.relu(self.fc2(fc1_out))
		#l1_penalty_fc2 = sum(p.abs().sum()
							 #for p in self.fc2.parameters())
		# Output
		w1_out = self.w1(fc2_out)
		l1_penalty_w1 = sum(p.abs().sum() for p in self.w1.parameters())
		w2_out = self.w2(fc2_out)
		l1_penalty_w2 = sum(p.abs().sum() for p in self.w2.parameters())

		#print(w1_out.shape)
		w_out = torch.stack([w1_out,w2_out],dim=1)
		#print(w_out.shape)
		select_out = self.select(fc2_out)
		#print("select_out", select_out.shape)
		#print("signature", self.signature.shape)
		score = F.softmax(torch.matmul(select_out, self.signature.transpose(1,0)),dim=1)
		score = score.unsqueeze(dim=2).repeat(1,1,w_out.shape[2])
		#print("score", score.shape)
		z = (w_out * score).sum(dim=1)
		#print("result", result.shape)
		#exit()



		l1_penalty = l1_penalty_w1 + l1_penalty_w2


		return z, l1_penalty


class Encoder_mlp(nn.Module):
	def __init__(self, args):
		super(Encoder_mlp, self).__init__()
		log.info('Building MLP encoder...')
		# Fully-connected layers
		log.info('FC layers...')
		self.fc1 = nn.Linear(32*32, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Flatten image
		x_flat = torch.flatten(x, 1)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(x_flat))
		fc2_out = self.relu(self.fc2(fc1_out))
		fc3_out = self.relu(self.fc3(fc2_out))
		# Output
		z = fc3_out
		return z

class Encoder_rand(nn.Module):
	def __init__(self, args):
		super(Encoder_rand, self).__init__()
		log.info('Building random encoder...')
		# Random projection
		self.fc1 = nn.Linear(32*32, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Flatten image
		x_flat = torch.flatten(x, 1)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(x_flat)).detach()
		# Output
		z = fc1_out.detach()
		return z


