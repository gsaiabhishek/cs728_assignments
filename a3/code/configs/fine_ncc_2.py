config = dict(
			model = dict(
						type = 'FFNModel',
						input_dim = 768*4, #depends on embed_dim, merge_type
						hidden_dim_1 = 2400,
						hidden_dim_2 = 2400,
						output_dim = 37, # 120 for fine cls, 37 for coarse cls # test
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
						name = 'fine_noun_compound_compositionability_2',
						trn_path = '../data/ncc/th/th.csv',
						embed_save_path = '../embed/th_bert_2.pkl', # test 
						merge_type = 'concat',
						embed_type = 'bert_pre_trained', # glove/bert_pre_trained
						embed_dim = 768, # 768 for bert
					)

		)