import torch
import torch.nn as nn
import torch.nn.functional as F
from util import log
import numpy as np
from tsf_modules import *
import math

def mirror(v):
	m =v.shape[-1]
	n=2*m-1
	w =torch.zeros(n)
	for t in range(n):
		if t < v.shape[0]:
			w[t]=v[t]
		else:
			w[t] = v[-(t-m+2)]
	return w

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return x

class Self_Attention(nn.Module):
	def __init__(self, in_dim, query_dim, dim, value_dim, nheads, sqrt, residual, norm_before, ffn_dim=128):
		super(Self_Attention, self).__init__()

		self.nheads = nheads
		self.head_dim = dim // nheads
		self.v_head_dim = value_dim // nheads
		self.ffn_dim = ffn_dim
		self.in_dim = in_dim
		self.query_dim = query_dim
		self.value_dim = value_dim
		self.dim = dim

		self.sqrt = sqrt
		self.residual = residual
		self.norm_before = norm_before




		self.query_net = nn.Linear(query_dim, dim)
		self.key_net = nn.Linear(in_dim, dim)
		self.value_net = nn.Linear(in_dim, value_dim)

		self.final = nn.Linear(value_dim, query_dim)

		self.res = nn.Sequential(
			nn.Linear(query_dim, ffn_dim),
			nn.Dropout(p=0.0),
			nn.ReLU(),
			nn.Linear(ffn_dim, query_dim),
			nn.Dropout(p=0.0)
		)
		self.norm1 = nn.LayerNorm(query_dim)
		self.norm2 = nn.LayerNorm(query_dim)

	def forward(self, query,x):
		# print("x shape", x.shape)
		# print("in_dim", self.in_dim,
		# 	  "query_dim", self.query_dim,
		# 	  "dim", self.dim,
		# 	  "value_dim", self.value_dim,
		# 	  "self.head_dim", self.head_dim)
		bsz, n, _ = x.shape

		#
		res = query
		if self.norm_before:
			query = self.norm1(query)

		q = self.query_net(query)
		q = q.reshape(bsz, 1, self.nheads, self.head_dim)
		if self.sqrt:
			q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)
		else:
			q = q.permute(0, 2, 1, 3)

		k = self.key_net(x).reshape(bsz, n, self.nheads, self.head_dim)
		k = k.permute(0,2,3,1)

		v = self.value_net(x).reshape(bsz, n, self.nheads, self.v_head_dim)
		v = v.permute(0,2,1,3)


		score = F.softmax(torch.matmul(q,k), dim=-1) # (bsz, nheads, n, n)
		out = torch.matmul(score, v) # (bsz, nheads, n, att_dim)
		out = out.view(bsz, self.nheads, 1, self.v_head_dim)
		out = out.permute(0, 2, 1, 3).reshape(bsz, 1, self.value_dim)
		if self.residual:
			out = self.final(out)
			if not self.norm_before:
				out = self.norm1(res + out)
			else:
				out = res + out

			res = out

			if self.norm_before:
				out = self.norm2(out)
				out = res + self.res(out)
			else:
				out = self.norm2(res + self.res(out))
			return out
		else:
			return out


class Model(nn.Module):
	def __init__(self, task_gen, args):
		super(Model, self).__init__()
		# Encoder
		log.info('Building encoder...')
		self.z_size = 128
		# self.gumbel = args.gumbel
		self.normalized = args.normalized
		self.key_dim = args.key_dim
		self.nheads = args.heads
		self.query_dim = args.query_dim
		self.value_dim = args.value_dim
		self.pos_dim = args.pos_dim
		self.layers = args.layers
		self.rules = args.rules
		self.residual = args.residual
		self.norm_before = args.norm_before
		self.sqrt = args.sqrt
		self.no_masking = args.no_masking
		self.reduced = args.reduced
		self.no_adding = args.no_adding
		self.no_positional = args.no_positional


		# self.hardcoded = args.hardcoded
		self.task = args.task
		# if self.task == "same_diff":
		# 	self.task_dim = 2
		# elif self.task == "RMTS":
		# 	self.task_dim = 6
		# else:
		# 	self.task_dim = 9

		self.task_dim = task_gen.seq_len


		self.detached = args.detached
		self.selfattention = args.selfattention
		if args.encoder == 'conv':
			self.shape_encoder = Encoder_conv(args)

		elif args.encoder == 'mlp':
			self.shape_encoder = Encoder_mlp(args)

		elif args.encoder == 'rand':
			self.shape_encoder = Encoder_rand(args)
		self.freeze_encoder = args.freeze_encoder
		if self.freeze_encoder:
			print("freezing")
			encoder_model = self.shape_encoder
			for param in encoder_model.parameters():
				print("freezing params")
				param.requires_grad = False
		# Building transformer
		log.info('Building Transformer and output layers...')


		if self.no_adding and self.reduced:
			self.matrix_size = (self.task_dim * (self.task_dim+1))//2
		else:
			self.matrix_size = self.task_dim*self.task_dim
		self.positional_vec = nn.Parameter(torch.randn(task_gen.seq_len, self.pos_dim))
		self.positional = nn.Parameter(torch.randn(self.matrix_size,2*self.pos_dim))
		self.pos_encoder = PositionalEncoding(self.pos_dim)
		self.query_vector = nn.Parameter(torch.randn(self.query_dim))
		self.query_matrix = nn.Linear(self.query_dim, self.key_dim)
		if self.no_positional:
			self.in_dim = self.rules
		else:
			self.in_dim = self.rules + 2 * self.pos_dim
		self.transformer = Self_Attention(self.pos_dim,
										  self.query_dim,
										  self.key_dim,
										  self.value_dim,
										  self.nheads,
										  self.sqrt,
										  self.residual,
										  self.norm_before)
		self.key_matrix = nn.Linear(self.in_dim, self.key_dim)
		self.value_matrix = nn.Linear(self.in_dim, self.value_dim)
		self.final = nn.Sequential(nn.Linear(self.query_dim, 256), nn.ReLU(),
								   nn.Linear(256, task_gen.y_dim))


		# Context normalization
		if args.norm_type == 'contextnorm' or args.norm_type == 'tasksegmented_contextnorm':
			self.contextnorm = True
			self.gamma = nn.Parameter(torch.ones(self.z_size))
			self.beta = nn.Parameter(torch.zeros(self.z_size))
		else:
			self.contextnorm = False
		if args.norm_type == 'tasksegmented_contextnorm':
			self.task_seg = task_gen.task_seg
		else:
			self.task_seg = [np.arange(task_gen.seq_len)]

	def forward(self, x_seq, device):
		# Encode all images in sequence
		s_seq = []
		# print("x_seq", x_seq.shape)
		for t in range(x_seq.shape[1]):
			x_t = x_seq[:,t,:,:,:]
			#print("x_t", x_t.shape)
			s_t = self.shape_encoder(x_t)
			s_seq.append(s_t)
		s_seq = torch.stack(s_seq, dim=1)
		if self.contextnorm:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				z_seq_all_seg.append(self.apply_context_norm(s_seq[:,self.task_seg[seg],:]))
			s_seq = torch.cat(z_seq_all_seg, dim=1)
		R_s = torch.matmul(s_seq, s_seq.transpose(2, 1))
		if self.selfattention:
			if self.no_masking:
				R_s = F.softmax(R_s, dim=2)
			else:
				mask_s = torch.ones_like(R_s[0]).triu(diagonal=1).unsqueeze(0)
				mask_s = mask_s.repeat(R_s.shape[0], 1, 1).bool()
				R_s.masked_fill_(mask_s, float('-inf'))
				R_s = F.softmax(R_s, dim=2)
				if not self.no_adding:
					R_s = R_s + R_s.tril(diagonal=-1).transpose(2,1)

		if self.reduced:
			C_s = torch.zeros_like(R_s)
			C_s[:, 0, :1] = R_s[:,0, :1]
			for i in range(1, R_s.shape[1]):
				C_s[:,i, :i + 1] = R_s[:, i, :i + 1]
		else:
			C_s = R_s



		R_save = R_s.clone()
		#print(C_s.shape)
		pos_vec = self.positional_vec.repeat(C_s.shape[0], 1, 1)
		pos_vec = torch.matmul(C_s, pos_vec)

		q_vector = self.query_vector
		q_vector = q_vector.repeat(R_s.shape[0], 1)
		q_vector = q_vector.reshape(R_s.shape[0], 1, self.query_dim)
		for _ in range(self.layers):
			q_vector = self.transformer(q_vector, pos_vec)
	
		y_pred_linear = self.final(q_vector.squeeze()).squeeze()
		y_pred = y_pred_linear.argmax(1)
		return y_pred_linear, y_pred, R_save
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq