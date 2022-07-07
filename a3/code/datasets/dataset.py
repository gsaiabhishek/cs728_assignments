from codecs import ignore_errors
from re import S
from matplotlib import lines
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from sklearn.model_selection import train_test_split
import json
import pickle
import os
import os.path as osp
import sys

def loadGloveModel(gloveFile):
	glove = pd.read_csv(gloveFile, sep=' ', header=None, encoding='utf-8', index_col=0, na_values=None, keep_default_na=False, quoting=3)
	return glove  # (word, embedding), 400k*dim

def get_unique_class_enum(cls):
	unique_cls = np.unique(np.array(cls))
	unique_cls = np.sort(unique_cls) # sorted and then enumerated to map labels to nums
	num_to_cls = {i : j for i, j in enumerate(unique_cls)}
	cls_to_num = {i : j for j, i in num_to_cls.items()}

	return num_to_cls, cls_to_num

def get_class_enum(cls):
	num_to_cls = {i : j for i, j in enumerate(cls)}
	cls_to_num = {i : j for j, i in num_to_cls.items()}
	return num_to_cls, cls_to_num

def count_pkl(path):
        if not osp.exists(path):
            return -1
        return_val = 0
        file = open(path, 'rb')
        while(True):
            try:
                _ = pickle.load(file)
                return_val += 1
            except EOFError:
                break
        file.close()
        return return_val

def get_hidden_states(encoded, token_ids_word, model, layers):
	'''Push input IDs through model. Stack and sum `layers` (last four by default).
	   Select only those subword token outputs that belong to our word of interest
	   and average them.'''
	with torch.no_grad():
		output = model(**encoded)
	states = output.hidden_states # Get all hidden states
	output = torch.stack([states[i] for i in layers]).sum(0).squeeze() # Stack and sum all requested layers
	word_tokens_output = output[token_ids_word] # Only select the tokens that constitute the requested word
	return word_tokens_output.mean(dim=0)


def get_word_vector(sent, idx, tokenizer, model, layers):
	'''Get a word vector by first tokenizing the input sentence, getting all token idxs
	   that make up the word of interest, and then `get_hidden_states`.'''
	encoded = tokenizer.encode_plus(sent, return_tensors='pt') # get all token idxs that belong to the word of interest
	token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
	return get_hidden_states(encoded, token_ids_word, model, layers)

def get_word_idx(sent: str, word: str):
	try:
		return (sent.lower()).split(" ").index(word.lower())
	except:
		return -1

def get_bert_embedding(sent, word, tokenizer, model, layers=None):
	# Use last four layers by default
	layers = [-4, -3, -2, -1] if layers is None else layers
	idx = get_word_idx(sent, word)
	word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)
	return word_embedding


def get_data(path, type):
	if type == 0: # for noun compound compositionability
		df = pd.read_csv(path, header = None)
		if len(df.columns) == 3:
			return [(i[0], i[1]) for i in np.array(df)], [float(i[2]) for i in np.array(df)]
		elif len(df.columns) == 2:
			return [(i[0], i[1]) for i in np.array(df)], [-1]*len(df) #note: -1 labels if absent
		else:
			print('Unknown no of columns in get_data for type 0')
			exit(1)
	elif type == 1: # for fine noun compound compositionability 1(indiviudal words embed) or 2(individual words and compound words embed)
		df = pd.read_csv(path, header = None)
		np_df = np.array(df)
		# np_df = np_df[:500, :] # test: including few data points

		# column 3 is coarse with 37 labels and column 4 is fine with 120 labels
		cls1 = np_df[:, 4] # fine class
		cls2 = np_df[:, 3] # coarse class # cls2 is used for labelling data in Dataset classes
		num_to_cls, cls_to_num = get_unique_class_enum(cls2)

		return [[i[0], i[1]] for i in np.array(np_df)], cls1, cls2, num_to_cls, cls_to_num 
	elif type == 2: # for fine noun compound compositionability 3(contextual words embed)
		file = open(path)
		data = json.load(file)
		words = []
		sents = []
		cls = []
		have_labels = False
		if 'label' in data[0]:
			have_labels = True
		for i in range(len(data)): # range(len(data)) # test: using small range
			word = data[i]['nc'].lower().split() # list of first and second word in noun compound
			for j in range(len(data[i]['context'])):
				if len(word) == 2 and get_word_idx(data[i]['context'][j], word[0]) >= 0 and \
					get_word_idx(data[i]['context'][j], word[1]) >= 0:
					
					words.append(word)
					sents.append(data[i]['context'][j].lower()) # lowering sent since planning to use bert-base-uncased
					if have_labels:
						cls.append(data[i]['label'])
					else:
						cls.append(-1)

		num_to_cls, cls_to_num = {}, {}
		if have_labels:
			num_to_cls, cls_to_num = get_unique_class_enum(cls)
		else:
			num_to_cls[-1] = -1
			cls_to_num[-1] = -1

		return words, sents, np.array(cls), num_to_cls, cls_to_num
	elif type == 3: # for trec ati from https://cogcomp.seas.upenn.edu/Data/QA/QC/
		lines = []
		with open(path, 'r', encoding='ISO-8859-1') as f:
			lines = f.readlines()
		coarse_labels, fine_labels, sents = [], [], []
		for line in lines:
			words = line.strip().split()
			sents.append(' '.join(words[1:]))
			coarse_labels.append(words[0].split(':')[0])
			fine_labels.append(words[0].split(':')[1])
		return coarse_labels, fine_labels, sents
	elif type == 4: # return all trec labels, https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html
		all_coarse = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
		all_fine = {'ABBR': ['abb', 'exp'], #2
					'ENTY': ['animal', 'body', 'color', 'cremat', 'currency', 'dismed', 'event', 'food', 'instru', 
							'lang', 'letter', 'other', 'plant', 'product', 'religion', 'sport', 'substance', 'symbol',
							'techmeth', 'termeq', 'veh', 'word'], #22
					'DESC': ['def', 'desc', 'manner', 'reason'], #4
					'HUM': ['gr', 'ind', 'title', 'desc'], #4
					'LOC': ['city', 'country', 'mount', 'other', 'state'], #5
					'NUM': ['code', 'count', 'date', 'dist', 'money', 'ord', 'other', 'period', 'perc', 'speed', 'temp',
							'volsize', 'weight']} #13
		num_to_coarse, coarse_to_num = get_class_enum(all_coarse)
		all_fine_flatten = []
		for k, v in all_fine.items():
			for fl in v:
				all_fine_flatten.append(k+':'+fl)
		num_to_fine, fine_to_num = get_class_enum(all_fine_flatten)
		return num_to_coarse, coarse_to_num, num_to_fine, fine_to_num

	else:
		print('Unknown df type in get_data')
		exit(1)

############## datasets start ############

class NCDataset(Dataset):
	def __init__(self, path, num_cls, embed_type = 'glove', wordvec = None, wordvec_dim = None, merge_type = 'concat', sigmoid_threshold = 0.5):
		data, labels = get_data(path, 0)
		self.merge_type = merge_type
		self.word1 = []
		self.word2 = []
		self.words = []
		self.relation = np.zeros(len(data))
		if embed_type == 'glove':
			for i in range(len(data)):
				self.words.append([data[i][0], data[i][1]])
				if data[i][0] in wordvec.index:
					self.word1.append(np.array(wordvec.loc[data[i][0]], dtype = float))
				else:
					self.word1.append(np.zeros(wordvec_dim, dtype = float))
				if data[i][1] in wordvec.index:
					self.word2.append(np.array(wordvec.loc[data[i][1]], dtype = float))
				else:
					self.word2.append(np.zeros(wordvec_dim, dtype = float))
		elif embed_type == 'bert_pre_trained':
			bert_type = 'bert-base-uncased'
			tokenizer = AutoTokenizer.from_pretrained(bert_type)
			model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

			for i in range(len(data)):
				self.words.append([data[i][0], data[i][1]])
				self.word1.append(np.array(get_bert_embedding(data[i][0]+' '+data[i][1], data[i][0], tokenizer, model), dtype = float))
				self.word2.append(np.array(get_bert_embedding(data[i][0]+' '+data[i][1], data[i][1], tokenizer, model), dtype = float))

		self.relation = np.array(labels)
		self.relation[self.relation > sigmoid_threshold] = 1
		self.relation[self.relation <= sigmoid_threshold] = 0

	def __getitem__(self, index):
		if self.merge_type == 'concat':
			return self.words[index], np.hstack([self.word1[index], self.word2[index]]), self.relation[index]
		elif self.merge_type == 'mean':
			return (self.word1[index]+self.word2[index])/2, self.relation[index]

	def __len__(self):
		return len(self.word1)


class Fine_NCDataset_0(Dataset):
	def __init__(self, data_path, wordvec, embed_dim):
		self.words, _, self.cls2, self.num_to_cls, self.cls_to_num = get_data(data_path, 1)
		
		self.word1, self.word2 = [], []
		self.num_cls = len(self.num_to_cls)

		for i in range(len(self.words)):
			if self.words[i][0] in wordvec.index:
				self.word1.append(np.array(wordvec.loc[self.words[i][0]], dtype = float))
			else:
				self.word1.append(np.zeros(embed_dim, dtype = float))

			if self.words[i][1] in wordvec.index:
				self.word2.append(np.array(wordvec.loc[self.words[i][1]], dtype = float))
			else:
				self.word2.append(np.zeros(embed_dim, dtype = float))

	def __getitem__(self, index):
		return self.words[index], np.hstack([self.word1[index], self.word2[index]]), self.cls_to_num[self.cls2[index]]

	def __len__(self):
		return len(self.word1)


class Fine_NCDataset_1_and_2(Dataset):
	def __init__(self, data_path, do_compound = False, embed_save_path = ''):
		self.words, _, self.cls2, self.num_to_cls, self.cls_to_num = get_data(data_path, 1)
		self.do_compound = do_compound # whether to include compound word embeddings in output or not
		
		self.word1, self.word2 = [], []
		self.word1_nc, self.word2_nc = [], []
		self.num_cls = len(self.num_to_cls)

		count_in_pkl = count_pkl(embed_save_path)
		if (self.do_compound and count_in_pkl == 5) or ((not self.do_compound) and count_in_pkl == 3):
			file_ = open(embed_save_path, 'rb')
			self.word1 = pickle.load(file_)
			self.word2 = pickle.load(file_)
			if self.do_compound:
				self.word1_nc = pickle.load(file_)
				self.word2_nc = pickle.load(file_)
			ignored_index = pickle.load(file_)
			file_.close()
		else:
			bert_type = 'bert-base-uncased'
			tokenizer = AutoTokenizer.from_pretrained(bert_type)
			model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

			count_nan = 0
			ignored_index = []
			for i in range(len(self.words)):
				if i % 1000 == 0:
					print('Fine_NCDataset:', i, 'number of words bert embeddings have been found')
				array1 = np.array(get_bert_embedding(self.words[i][0], self.words[i][0], tokenizer, model), dtype = float)
				array2 = np.array(get_bert_embedding(self.words[i][1], self.words[i][1], tokenizer, model), dtype = float)
				if self.do_compound:
					array3 = np.array(get_bert_embedding(self.words[i][0]+' '+self.words[i][1], self.words[i][0], tokenizer, model), dtype = float)
					array4 = np.array(get_bert_embedding(self.words[i][0]+' '+self.words[i][1], self.words[i][1], tokenizer, model), dtype = float)
				
				is_nan1 = np.isnan(array1).any()
				is_nan2 = np.isnan(array2).any()
				if not is_nan1 and not is_nan2:
					if do_compound:
						is_nan3 = np.isnan(array3).any()
						is_nan4 = np.isnan(array4).any()
						if not is_nan3 and not is_nan4:
							self.word1.append(array1)
							self.word2.append(array2)
							self.word1_nc.append(array3)
							self.word2_nc.append(array4)
						else:
							count_nan += 1
							ignored_index.append(i)
					else:
						self.word1.append(array1)
						self.word2.append(array2)
				else:
					count_nan += 1
					ignored_index.append(i)

			print('No of NaN bert embeds ignored in Fine_NCDataset_1_and_2:', count_nan)

			file_ = open(embed_save_path, 'wb')
			pickle.dump(self.word1, file_)
			pickle.dump(self.word2, file_)
			if self.do_compound:
				pickle.dump(self.word1_nc, file_)
				pickle.dump(self.word2_nc, file_)
			pickle.dump(ignored_index, file_)
			file_.close()
		
		self.cls2 = np.delete(self.cls2, ignored_index)
		for i in sorted(ignored_index, reverse=True):
			del self.words[i]

	def __getitem__(self, index):
		if self.do_compound:
			return self.words[index], np.hstack([self.word1[index], self.word2[index], self.word1_nc[index], self.word2_nc[index]]), self.cls_to_num[self.cls2[index]]
		else:
			return self.words[index], np.hstack([self.word1[index], self.word2[index]]), self.cls_to_num[self.cls2[index]]

	def __len__(self):
		return len(self.word1)


class Fine_NCDataset_3(Dataset):
	def __init__(self, data_path, embed_save_path = ''):
		self.words, self.sents, self.cls, self.num_to_cls, self.cls_to_num = get_data(data_path, 2)
		
		self.word1, self.word2 = [], []
		self.num_cls = len(self.num_to_cls)

		if count_pkl(embed_save_path) == 3:
			file_ = open(embed_save_path, 'rb')
			self.word1 = pickle.load(file_)
			self.word2 = pickle.load(file_)
			ignored_index = pickle.load(file_)
			file_.close()
		else:
			bert_type = 'bert-base-uncased'
			tokenizer = AutoTokenizer.from_pretrained(bert_type)
			model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

			bert_exceptions = 0
			count_nan = 0
			ignored_index = []
			for i in range(len(self.words)):
				if i % 10000 == 0:
					print('Fine_NCDataset:', i, 'number of words bert embeddings have been found')
					print(str(i)+' Done out of '+str(len(self.words)), file=sys.stderr)
				try:
					array1 = np.array(get_bert_embedding(self.sents[i], self.words[i][0], tokenizer, model), dtype = float)
					array2 = np.array(get_bert_embedding(self.sents[i], self.words[i][1], tokenizer, model), dtype = float)
					is_nan1 = np.isnan(array1).any()
					is_nan2 = np.isnan(array2).any()
					if not is_nan1 and not is_nan2:
						self.word1.append(array1)
						self.word2.append(array2)
					else:
						count_nan += 1
						ignored_index.append(i)
				except:
					bert_exceptions += 1
					ignored_index.append(i)
			print('Bert Exception count in Fine_NCDataset_3:', bert_exceptions)
			print('No of NaN bert embeds ignored in Fine_NCDataset_3:', count_nan)

			file_ = open(embed_save_path, 'wb')
			pickle.dump(self.word1, file_)
			pickle.dump(self.word2, file_)
			pickle.dump(ignored_index, file_)
			file_.close()
		
		self.cls = np.delete(self.cls, ignored_index)
		for i in sorted(ignored_index, reverse=True):
			del self.words[i]

	def __getitem__(self, index):
		return self.words[index], np.hstack([self.word1[index], self.word2[index]]), self.cls_to_num[self.cls[index]]

	def __len__(self):
		return len(self.word1)


class Trec_Dataset(Dataset):
	def __init__(self, data_path, max_len, token_save_path = '', is_coarse = True):
		self.coarse_label, self.fine_label, self.sents = get_data(data_path, 3)
		self.is_corase = is_coarse
		self.num_to_coarse, self.coarse_to_num, self.num_to_fine, self.fine_to_num = get_data('', 4)
		self.tokeniser = BertTokenizer.from_pretrained('bert-base-cased')
		self.max_len = max_len

		if count_pkl(token_save_path) == 2:
			file_ = open(token_save_path, 'rb')
			self.all_ids = pickle.load(file_)
			self.all_masks = pickle.load(file_)
			file_.close()
		else:
			self.all_ids, self.all_masks = [], []
			for index in range(len(self.sents)):
				inputs = self.tokeniser.encode_plus(self.sents[index], None, add_special_tokens=True, max_length = self.max_len,
					padding = 'max_length', truncation = True)
				ids = inputs['input_ids']
				mask = inputs['attention_mask']
				padding_length = self.max_len - len(ids)
				ids = ids + ([0] * padding_length)
				mask = mask + ([0] * padding_length)
				self.all_ids.append(ids)
				self.all_masks.append(mask)
			file_ = open(token_save_path, 'wb')
			pickle.dump(self.all_ids, file_)
			pickle.dump(self.all_masks, file_)
			file_.close()

		print('Trec_Dataset init done')

	def __getitem__(self, index):
		if self.is_corase:
			label = self.coarse_to_num[self.coarse_label[index]]
		else:
			label = self.fine_to_num[self.coarse_label[index]+':'+self.fine_label[index]]

		return self.sents[index], {'ids': torch.tensor(self.all_ids[index], dtype=torch.long), 'mask': torch.tensor(self.all_masks[index], dtype=torch.long)}, \
			torch.tensor(label, dtype=torch.long)

	def __len__(self):
		return len(self.sents)



############## datasets end ############


def get_dataset(info):
	if info.name == 'noun_compound_compositionability':
		num_cls = 2

		wordvec = None
		if info.embed_type == 'glove':
			weight_full_path = info.glove_path+'glove.6B.' + str(info.embed_dim) + 'd.txt'
			wordvec = loadGloveModel(weight_full_path)
		elif info.embed_type == 'bert_pre_trained':
			pass
		else:
			print('Unknown embed type in noun_compound_compositionability in get_dataset()')
			exit(1)

		merge_type = info.merge_type
		trn_threshold = 0.5
		test_threshold = info.sigmoid_threshold

		trn_set = NCDataset(info.trn_path, num_cls, info.embed_type, wordvec, info.embed_dim, merge_type, trn_threshold)
		test_sets = [NCDataset(info.test_1_path, num_cls, info.embed_type, wordvec, info.embed_dim, merge_type, test_threshold), \
			NCDataset(info.test_2_path, num_cls, info.embed_type, wordvec, info.embed_dim, merge_type, test_threshold)]
		return trn_set, None, test_sets, num_cls
	elif info.name == 'fine_noun_compound_compositionability_0': # 768*2 embed dim
		weight_full_path = info.glove_path+'glove.6B.' + str(info.embed_dim) + 'd.txt'
		wordvec = loadGloveModel(weight_full_path)

		data_set = Fine_NCDataset_0(info.trn_path, wordvec, info.embed_dim)
		num_cls = data_set.num_cls

		targets = [data_set.cls_to_num[i] for i in data_set.cls2]
		trn_idx, test_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state = 143, shuffle=True, stratify=targets)
		trn_set = Subset(data_set, trn_idx)
		test_set = Subset(data_set, test_idx)

		return trn_set, None, [test_set], num_cls
	elif info.name == 'fine_noun_compound_compositionability_1': # 768*2 embed dim
		data_set = Fine_NCDataset_1_and_2(info.trn_path, False, info.embed_save_path)
		num_cls = data_set.num_cls

		targets = [data_set.cls_to_num[i] for i in data_set.cls2]
		trn_idx, test_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state = 143, shuffle=True, stratify=targets)
		trn_set = Subset(data_set, trn_idx)
		test_set = Subset(data_set, test_idx)

		return trn_set, None, [test_set], num_cls
	elif info.name == 'fine_noun_compound_compositionability_2': # 768*4 embed dim
		data_set = Fine_NCDataset_1_and_2(info.trn_path, True, info.embed_save_path)
		num_cls = data_set.num_cls

		targets = [data_set.cls_to_num[i] for i in data_set.cls2]
		trn_idx, test_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state = 143, shuffle=True, stratify=targets)
		trn_set = Subset(data_set, trn_idx)
		test_set = Subset(data_set, test_idx)

		return trn_set, None, [test_set], num_cls
	elif info.name == 'fine_noun_compound_compositionability_3': # 768*2 embed dim
		trn_set = Fine_NCDataset_3(info.trn_path, info.trn_embed_save_path)
		test_set = Fine_NCDataset_3(info.test_path, info.test_embed_save_path)
		num_cls = trn_set.num_cls

		return trn_set, None, [test_set], num_cls
	elif info.name == 'trec_ati':
		trn_set = Trec_Dataset(info.trn_path, info.max_len, info.token_trn_save_path, info.is_coarse)
		test_set = Trec_Dataset(info.test_path, info.max_len, info.token_test_save_path, info.is_coarse)
		num_cls = 6 if info.is_coarse else 50

		return trn_set, None, [test_set], num_cls
	else:
		return
