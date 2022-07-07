from numpy import result_type
from utils import load_config_file
from train import Trainer
import pandas as pd
from datasets import get_data

if __name__ == '__main__':

	expmnt_type = 3 # 0,1,2,3
	run_num = 1
	cfg_file = 'configs/fine_ncc_'
	results_folder = '../results/'
	cfg_file = cfg_file+str(expmnt_type)+'.py'

	result_type = ''
	have_targets = True
	get_data_type = 1

	if expmnt_type == 0:
		result_type = '0_0_0'
	elif expmnt_type == 1:
		result_type = '1_0_0'
	elif expmnt_type == 2:
		result_type = '1_1_0'
	elif expmnt_type == 3:
		result_type = '0_0_1'
		get_data_type = 2
		have_targets = False
	else:
		print('Unknown experiment type')
		exit(1)

	cfg = load_config_file(cfg_file)
	
	print('Experiment type:', result_type)
	print('Note: convention is <1/0>_<1/0>_<1/0>. Where 1/0 corresponds to indiviudal word bert \
embed, compound word bert embed, contextual sentences bert embed used or not respectively. all 0\'s mean glove embed')

	trainer = Trainer(cfg)
	max_acces = trainer.train()
	print('Best Train Acc: '+str(max_acces['trn']))
	print('Best Test Acc: '+str(max_acces['test0']))
	trn_loss, _, _, trn_acc, trn_words_labels = trainer.evaluate('trn', -1)
	test_loss, _, _, test_acc, test_words_labels = trainer.evaluate('test', -1, 0, have_targets = have_targets)

	_, _, _, num_to_cls, cls_to_num = get_data(cfg.dataset.trn_path, get_data_type)
	for i in range(len(trn_words_labels)):
		trn_words_labels[i][2] = num_to_cls[trn_words_labels[i][2]]
	for i in range(len(test_words_labels)):
		test_words_labels[i][2] = num_to_cls[test_words_labels[i][2]]

	result_file_trn = results_folder+result_type+'_trn_'+str(run_num)+'.csv'
	result_file_test = results_folder+result_type+'_test_'+str(run_num)+'.csv'

	df0 = pd.DataFrame(trn_words_labels, index = None, columns = None)
	df1 = pd.DataFrame(test_words_labels, index = None, columns = None)
	df0.to_csv(result_file_trn, index = False, header = False)
	df1.to_csv(result_file_test, index = False, header = False)



