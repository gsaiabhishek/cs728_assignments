config = dict(
			model = dict(
						type = 'FFNModel',
						input_dim = 1536, #depends on embed_dim, merge_type
						hidden_dim_1 = 1000,
						hidden_dim_2 = 1000,
						output_dim = 1,
					),
			loss = dict(
						type = 'BCELoss',
					),
			optimizer=dict(
						type = "adam",
						momentum = 0.9,
						lr = 0.001,
						weight_decay = 5e-4
					),
			scheduler=dict(
						type=None,
						#type="cosine_annealing",
						T_max=300
					),
			train_args = dict(
						num_epochs = 20,
						device = 'cpu',
						print_every = 2,
						print_args = ['trn_acc', 'trn_loss', 'test_acc', 'test_loss']
					),
			dataloader = dict(
						trn_bs = 64, # batch_size
						val_bs = 64,
						test_bs = 64,
						shuffle = True,
					),
			dataset = dict(
						name = 'noun_compound_compositionability',
						trn_path = '../data/ncc/f/f.csv',
						test_1_path = '../data/ncc/r1/r1.csv',
						test_2_path = '../data/ncc/r2/r2.csv',
						merge_type = 'concat',
						embed_type = 'bert_pre_trained', # glove/bert_pre_trained
						glove_path = '../data/glove.6B/',
						embed_dim = 768, # 300 for glove, 768 for bert
						sigmoid_threshold = 0.5,
					)

		)