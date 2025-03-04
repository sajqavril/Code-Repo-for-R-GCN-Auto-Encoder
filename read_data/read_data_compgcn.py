# from ..helper import *
from .data_loader import TrainDataset, TestDataset
import gzip
import random
import math
import torch
from ordered_set import OrderedSet
from collections import defaultdict as ddict
from torch.utils.data import DataLoader
from helper import *
import pickle

class read_compgcn(object):

	def __init__(self, params):
		self.p = params
		self.triplet_no_reverse = self.p.triplet_no_reverse
		# edge_index, edge_type = self.load_data()


	def load_data(self):
		# if self.p.dataset in ["NELL-995", "WN18RR", "FB15k-237"]:

		# 	if self.p.dataset == "NELL-995":
		# 		self.p.num_ent = 75492
		# 		self.p.num_rel = 200
		# 		self.p.num_ent_type = 267
		# 	else:	
		# 		for name in ['entity', 'relation']:
		# 			for line in open(self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, name)):
		# 				if name == 'entity':
		# 					self.p.num_ent		= int(line.strip())
		# 				if name == 'relation':
		# 					self.p.num_rel		= int(line.strip())
		# 				break
		# 		for line in open(self.p.data_dir+'/{}/type_constrain.txt'.format(self.p.dataset)):
		# 			self.p.num_ent_type = int(line.strip())
		# 			break

		# 	self.triples  = ddict(list)
		# 	self.data = {}
		# 	sr2o = ddict(set)
		# 	train_edge_num  = 0
		# 	unique_train = set()
		# 	if self.p.type_noise > 0.:
		# 		file_name = {
		# 			'train': self.p.data_dir+'/{}/train_ind_true_type_noise_type_{:.2f}_all.txt'.format(self.p.dataset, self.p.type_noise),
		# 			'valid': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'valid'),
		# 			'test': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'test')
		# 		}
		# 	elif self.p.normal_noise > 0.:
		# 		file_name = {
		# 			'train': self.p.data_dir+'/{}/train_ind_true_type_noise_{:.2f}_all.txt'.format(self.p.dataset, self.p.normal_noise),
		# 			'valid': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'valid'),
		# 			'test': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'test')
		# 		}
		# 	else:
		# 		file_name = {
		# 			'train': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'train'),
		# 			'valid': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'valid'),
		# 			'test': self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, 'test')
		# 		}


		# 	for split in ['train', 'test', 'valid']:
		# 		self.data[split] = []
		# 		for line in open(file_name[split]):
		# 			try:
		# 				sub, obj, rel = line.strip().split(' ')
		# 				sub = int(sub.strip())
		# 				rel = int(rel.strip())
		# 				obj = int(obj.strip())
			 
		# 			except:
		# 				continue
		# 			self.data[split].append((sub, rel, obj))
					
		# 			if split == 'train': 

		# 				train_edge_num += 1
		# 				unique_train.add(sub)
		# 				unique_train.add(obj)

		# 				sr2o[(sub, rel)].add(obj) 
		# 				if self.triplet_no_reverse:
		# 					sr2o[(obj, rel)].add(sub)	
		# 				else:
		# 					sr2o[(obj, rel+self.p.num_rel)].add(sub)
					

		# 	self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		# 	self.entity_type = torch.zeros(size=(self.p.num_ent, ))

		# 	i = 0
		# 	for line in open(self.p.data_dir+'/{}/type_constrain.txt'.format(self.p.dataset)):
		# 		if i==0:
		# 			i = i + 1
		# 			continue
		# 		try:
		# 			nodes = [int(j) for j in line.split('\t')]
		# 			typ = int(nodes[0])
		# 			self.entity_type[torch.LongTensor(nodes[1:])] = typ
		# 		except:
		# 			continue
		# 	self.entity_type = self.entity_type.to(torch.long)


		# 	self.sr2o = {k: list(v) for k, v in sr2o.items()}
		# 	for split in ['test', 'valid']:
		# 		for sub, rel, obj in self.data[split]:
		# 			sr2o[(sub, rel)].add(obj)

		# 			if self.triplet_no_reverse:
		# 				sr2o[(obj, rel)].add(sub)
		# 			else:
		# 				sr2o[(obj, rel+self.p.num_rel)].add(sub)


		# 	self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		# 	self.triples  = ddict(list)

		# 	rel_number = self.p.num_rel

	
		ent_set, rel_set, ent_type_set = OrderedSet(), OrderedSet(), OrderedSet()

		self.ent2type = {}
		
	
		for split in ['train', 'test', 'valid']:
			for line in open(self.p.data_dir+'/{}/{}.txt'.format(self.p.dataset, split)):
				# print(line.strip().split('\t'))
				if len(line.strip().split('\t')) < 3:
					
					continue
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

				if self.p.dataset == "NELL-995":
					self.ent2type[sub] = sub.split('_')[1].strip()
					ent_type_set.add(sub.split('_')[1].strip())
					try:
						ent_type_set.add(obj.split('_')[1].strip())
						self.ent2type[obj] = sub.split('_')[1].strip()
					except:
						ent_type_set.add('concept')
						self.ent2type[obj] = 'concept'
		
		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		if self.p.dataset == "NELL-995":
			self.p.num_ent = 75492
			self.p.num_rel = 200
			self.p.num_ent_type = 267
		else:	
			for name in ['entity', 'relation']:
				for line in open(self.p.data_dir+'/{}/{}2id.txt'.format(self.p.dataset, name)):
					if name == 'entity':
						self.p.num_ent		= int(line.strip())
					if name == 'relation':
						self.p.num_rel		= int(line.strip())
					break
			for line in open(self.p.data_dir+'/{}/type_constrain.txt'.format(self.p.dataset)):
				self.p.num_ent_type = int(line.strip())
				break

		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		self.entity_type = torch.zeros(size=(self.p.num_ent, ))

		i = 0
		for line in open(self.p.data_dir+'/{}/type_constrain.txt'.format(self.p.dataset)):
			if i==0:
				i = i + 1
				continue
			try:
				nodes = [int(j) for j in line.split('\t')]
				typ = int(nodes[0])
				self.entity_type[torch.LongTensor(nodes[1:])] = typ
			except:
				continue
		self.entity_type = self.entity_type.to(torch.long)

		print('dataset: ', self.p.dataset)
		print('number of entities: ', self.p.num_ent)
		print('number of relations: ', self.p.num_rel)
		

		self.data = ddict(list)
		sr2o = ddict(set)
		train_edge_num = 0
		unique_train = set()
		for split in ['train', 'test', 'valid']:
			for line in open(self.p.data_dir+'/{}/{}.txt'.format(self.p.dataset, split)):
				if len(line.strip().split('\t')) < 3:
					continue
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
		
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
			

				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					train_edge_num += 1
					unique_train.add(sub)
					unique_train.add(obj)

					sr2o[(sub, rel)].add(obj) 

					if not self.triplet_no_reverse:
						# print('use inverse in loss')
						sr2o[(obj, rel+self.p.num_rel)].add(sub)
					else:
						# print('do not use inverse in loss')
						sr2o[(obj, rel)].add(sub)


		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)

				if not self.triplet_no_reverse:
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
				else:
					sr2o[(obj, rel)].add(sub)


		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		self.triples  = ddict(list)


		for (sub, rel), obj in self.sr2o.items():
			# num_label += len(self.sr2o[(sub, rel)])
			self.triples['train'].append({'triple':(sub, rel, obj[0]), 'label': self.sr2o[(sub, rel)],'sub_samp': 1})
		
		
		for split in ['test', 'valid']:
		
			for sub, rel, obj in self.data[split]:
				if not self.triplet_no_reverse:
					rel_inv = rel + self.p.num_rel
				else:
					rel_inv = rel
			
				
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split],  self.p.num_ent, 0, False, self.p.lbl_smooth),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			# 'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			# 'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			# 'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			# 'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}
		self.p.neg_num = 0

		# if self.p.less_edges_in_aggre:
		# 	self.edge_index, self.edge_type = less_edge_index, less_edge_type
			
		# else:
		rel_number = self.p.num_rel
		
		self.edge_index, self.edge_type = construct_adj(self.data, self.p.num_ent, rel_number, self.p.noise_rate, self.p.all_noise, self.p.data_dir, self.p.dataset)
		
		
		################## use fixed features

		file = self.p.data_dir + '/'+self.p.dataset+'/2hop_neighbor_myindex.pickle'
		
		if os.path.exists(file):
			with open(file, 'rb') as handle:
				node_neighbors = pickle.load(handle)
				
			indices_2hop = read_neighbor_2hop(node_neighbors,  self.ent2id, self.rel2id, unique_train, self.p)
		else:
			indices_2hop = None

		if self.p.use_feat_input:
			print('use transe feature!!')
			file = self.p.data_dir + '/'+self.p.dataset+'/feature_embedding.pickle'

			with open(file, 'rb') as handle:
				embedding_dict = pickle.load(handle)
			
			feature_embeddings=read_feature(embedding_dict, self.ent2id, self.rel2id)

		# elif self.p.use_type_feat:
		# 	feature_embeddings = torch.eye(self.entity_type.max() + 1).to(torch.float)[self.entity_type]

		else:
			print('use random feature!!')
			feature_embeddings = None
		
		
		if not self.p.no_edge_reverse:
			
			print('max rel index: ', max(self.edge_type))
			return self.edge_index, self.edge_type, self.data_iter, feature_embeddings, indices_2hop
			
		else:
			
			e = self.edge_index.size(-1) // 2
			print('edge no inverse:', self.edge_index[:, :e].size(), max(self.edge_type[:e]))
			return self.edge_index[:, :e], self.edge_type[:e], self.data_iter, feature_embeddings, indices_2hop
			
		


		# return self.edge_index, self.edge_type, self.data_iter, feature_embeddings

	def read_batch(self, batch, split, device):
		
		if split == 'train':
			triple, label, pos_neg_ent  = [ _.to(device) for _ in batch]
			pos_neg_ent = None
			
			return triple[:, 0], triple[:, 1], triple[:, 2], label, pos_neg_ent
			
		else:
			triple, label = [ _.to(device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
	

