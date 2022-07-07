import torch

config = dict(
			model = dict(
						type = 'BertCased',
						bert_dim = 768,
						output_dim = 50,
						dropout = 0.3
					),
			loss = dict(
						type = 'CrossEntropyLoss',
					),
			optimizer=dict(
						type = "adam",
						momentum = 0.9,
						lr = 1e-6,
						weight_decay = 5e-4
					),
			scheduler=dict(
						type=None,
						#type="cosine_annealing",
						T_max=300
					),
			train_args = dict(
						num_epochs = 25,
						device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
						print_every = 1,
						parse_type = 1,
						print_args = ['trn_acc', 'trn_loss', 'test_acc', 'test_loss']
					),
			dataloader = dict(
						trn_bs = 8, # batch_size
						val_bs = 8,
						test_bs = 8,
						shuffle = True,
					),
			dataset = dict(
						name = 'trec_ati',
						trn_path = '../data/train_5500.label.txt',
                        test_path = '../data/TREC_10.label.txt',
						token_trn_save_path = './token/trn_1.pkl',
						token_test_save_path = './token/test_1.pkl',
						is_coarse = False,
						max_len = 128
					)

		)