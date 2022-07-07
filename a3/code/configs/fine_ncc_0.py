config = dict(
			model = dict(
						type = 'FFNModel',
						input_dim = 300*2, #depends on embed_dim, merge_type
						hidden_dim_1 = 480,
						hidden_dim_2 = 480,
						output_dim = 37,
					),
			loss = dict(
						type = 'CrossEntropyLoss',
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
						num_epochs = 25,
						device = 'cpu',
						print_every = 2,
						print_args = ['trn_acc', 'trn_loss', 'test_acc', 'test_loss']
					),
			dataloader = dict(
						trn_bs = 128, # batch_size
						val_bs = 128,
						test_bs = 128,
						shuffle = True,
					),
			dataset = dict(
						name = 'fine_noun_compound_compositionability_0',
						trn_path = '../data/ncc/th/th.csv',
						merge_type = 'concat',
						embed_type = 'glove', # glove/bert_pre_trained
						glove_path = '../data/glove.6B/',
						embed_dim = 300, # 300 for glove, 768 for bert
					)

		)