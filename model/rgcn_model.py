from helper import *

from torch import nn
import torch.nn.functional as F
from model.rgcn_conv import RGCNConv
from model.torch_rgcn_conv import FastRGCNConv

class RGCNModel(torch.nn.Module):
	def __init__(self, edge_index, edge_type, params, node_type=None):
		super(RGCNModel, self).__init__()

		self.p = params
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.node_type = node_type

		self.act	= torch.tanh
		num_rel = self.p.num_rel

		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		if self.p.use_type_feat:
			self.type_embed = get_param((self.p.num_ent_type,   self.p.init_dim))
			# self.init_embed = self.type_embed[self.node_type]
		else:
			self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))

		# self.init_rel = get_param(( edge_type.unique().size(0), self.p.embed_dim))

		self.init_rel = get_param(( num_rel*2, self.p.init_dim))
		
		self.w_rel 		= get_param((self.p.init_dim, self.p.embed_dim))

		self.drop = torch.nn.Dropout(self.p.hid_drop)
		
		self.convs = nn.ModuleList()

		if self.p.gcn_layer == 1: self.act = None
		self.rgcn_conv1 = RGCNConv(self.p.init_dim, self.p.gcn_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks, act=self.act)
		self.convs.append(self.rgcn_conv1)

		for i in range(1, self.p.gcn_layer):
			conv = RGCNConv(self.p.embed_dim, self.p.embed_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks) #if self.p.gcn_layer == 2 else None
			self.convs.append(conv)
	

	def forward(self, feature=None, edge_norm=None):
		
		self.edge_index = self.edge_index.to(self.init_rel.device)
		self.edge_type = self.edge_type.to(self.init_rel.device)
		
		if self.p.use_type_feat:
			x = self.type_embed.to(self.init_rel.device)[self.node_type.to(self.init_rel.device)]
		else:
			x = self.init_embed.to(self.init_rel.device)
		r = self.init_rel

		if feature != None:
			
			x = feature['entity_embedding']
			x = x.to(self.init_rel.device)
			
			if 'relation_embedding' in feature:
				r = Parameter(feature['relation_embedding'])
				r = r.to(self.init_rel.device)
				
		for conv in self.convs:
			x = conv(x, self.edge_index, self.edge_type, edge_norm)
			x = self.drop(x)

		r = torch.matmul(r, self.w_rel)
		# print('rel weight:', self.w_rel)
		# print('rel weight norm', torch.norm(self.w_rel, p=2), (self.w_rel).mean(), self.w_rel.std())
		
		return x, r



class RGCNAEModel(torch.nn.Module):
	def __init__(self, edge_index, edge_type, node_type, params):
		super(RGCNAEModel, self).__init__()

		self.p = params
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.node_type = node_type

		# enoder
		self.encoder = RGCNModel(edge_index=self.edge_index,
						  		edge_type=self.edge_type,
								node_type=self.node_type,
								params=self.p)
		
		self.enc_lins = torch.nn.ModuleList()
		self.enc_lins.append(nn.Linear((self.p.embed_dim + self.p.embed_dim + self.p.embed_dim) , self.p.embed_dim, bias=True))
		self.enc_lins.append(nn.Linear(self.p.embed_dim, 1, bias=True))
		
		self.enc_lins[0].bias.data = torch.ones((self.p.embed_dim))
		self.enc_lins[1].bias.data = torch.ones((1))
		

		# decoder
		self.decoder = RGCNModel(edge_index=self.edge_index,
						  		edge_type=self.edge_type,
								node_type=self.node_type,
								params=self.p)
		
		self.dec_lins = nn.ModuleList()
		self.dec_lins.append(nn.Linear((self.p.embed_dim + self.p.embed_dim + self.p.embed_dim), self.p.embed_dim, bias=True))
		self.dec_lins.append(nn.Linear(self.p.embed_dim, self.p.num_rel, bias=True))

	def mlp_score(self, h, r, t, lins):

		h_ = torch.cat([h, r, t], dim=1)
		h_= F.normalize(h_)
		h_ = F.relu(lins[0](h_))
		score = lins[1](h_)
		
		return score.sigmoid()

		

	def gumbel_softmax(self, pi_pos, temparature=0.1):

		# pi_pos = pi_pos
		# pi_neg = (1.-pi_pos)
		# ui_pos = torch.distributions.Uniform(low=0., high=1.).sample((pi_pos.shape[0], )).to(pi_pos.device)
		# ui_neg = torch.distributions.Uniform(low=0., high=1.).sample((pi_pos.shape[0], )).to(pi_pos.device)
		# gi_pos = (-1.) * torch.log10((-1)*torch.log10(ui_pos))
		# gi_neg = (-1.) * torch.log10((-1)*torch.log10(ui_neg))
		# exp_log_pi_gi_pos = torch.exp((torch.log(pi_pos) + gi_pos) / temparature)
		# exp_log_pi_gi_neg = torch.exp((torch.log(pi_neg) + gi_neg) / temparature)
		# exp_log_pi_gi_sum = exp_log_pi_gi_pos + exp_log_pi_gi_neg
		# gumbel_softmax = exp_log_pi_gi_pos / exp_log_pi_gi_sum

		pi_neg = 1. - pi_pos
		pi = torch.stack([pi_pos, pi_neg], dim=0).requires_grad_()
		gsm_score = torch.nn.functional.gumbel_softmax(pi, tau=0.1, hard=True, dim=0)[0]

		return gsm_score


	def forward(self, feature=None):
				
		# encoder 
		x, r = self.encoder(feature=None, edge_norm=None)
		score = self.mlp_score(h=x[self.edge_index[0]],
						 	   r=r[self.edge_type],
							   t=x[self.edge_index[1]],
							   lins=self.enc_lins)
		gsm_score = self.gumbel_softmax(score)
		if self.p.no_gumbel:
			edge_norm = score
		else:
			edge_norm = gsm_score

		# decoder
		self.gsm_score = gsm_score.detach()
		self.score = score.detach()
		x_, r_ = self.decoder(edge_norm=edge_norm)

		# pred_link = self.mlp_score(h=x_[self.edge_index[0]],
		# 				 	   r=r_[self.edge_type],
		# 					   t=x_[self.edge_index[1]],
		# 					   lins=self.dec_lins) # this batch of scores should be the highest
		
		return x_, r_, gsm_score, score

