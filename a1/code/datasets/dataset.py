import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

############## datasets start ############

class NCDataset(Dataset):
	def __init__(self, path, num_cls, embed_type = 'glove', wordvec = None, wordvec_dim = None, merge_type = 'concat', sigmoid_threshold = 0.5):
		data, labels = get_csv_data(path, 0)
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
				self.word1.append(np.array(get_bert_embedding(data[i][0]+' '+data[i][1], 0, tokenizer, model), dtype = float))
				self.word2.append(np.array(get_bert_embedding(data[i][0]+' '+data[i][1], 1, tokenizer, model), dtype = float))

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

############## datasets end ############

def loadGloveModel(gloveFile):
	glove = pd.read_csv(gloveFile, sep=' ', header=None, encoding='utf-8', index_col=0, na_values=None, keep_default_na=False, quoting=3)
	return glove  # (word, embedding), 400k*dim

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


def get_bert_embedding(sent, idx, tokenizer, model, layers=None):
	# Use last four layers by default
	layers = [-4, -3, -2, -1] if layers is None else layers
	word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)
	return word_embedding


def get_csv_data(path, type):
	df = pd.read_csv(path, header = None)
	if type == 0: # for noun compound compositionability
		if len(df.columns) == 3:
			return [(i[0], i[1]) for i in np.array(df)], [float(i[2]) for i in np.array(df)]
		elif len(df.columns) == 2:
			return [(i[0], i[1]) for i in np.array(df)], [-1]*len(df) #note: -1 labels if absent
		else:
			print('Unknown df type in get_csv_data')
			exit(1)

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
	else:
		return
