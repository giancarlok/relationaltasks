import torch
import torch.nn as nn
import torch.nn.functional as F
from util import log
import numpy as np
from tsf_modules import *
import math


from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'



class GCN_module(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN_module, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, adj)
		return F.log_softmax(x, dim=1)


#################################

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
		self.task_dim = task_gen.seq_len
		self.gcn = GCN_module(128, 128, self.task_dim, 0.5)

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

		self.final = nn.Sequential(nn.Linear(self.matrix_size, 256), nn.ReLU(), nn.Linear(256,task_gen.y_dim))


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
		adj_matrix = torch.Tensor(np.ones((self.task_dim, self.task_dim), dtype=int))
		results = []
		linear_results = torch.empty(0,self.task_dim).to(device)
		for i in range(s_seq.shape[0]):
			result = self.gcn(s_seq[i].to(device), adj_matrix.to(device))[0]
			results.append(result.argmax(0))
			linear_results = torch.cat([linear_results, result[None,:]], dim=0)

		y_pred = torch.Tensor(results).to(device)
		y_pred_linear = linear_results.to(device)

		return y_pred_linear, y_pred, None
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq