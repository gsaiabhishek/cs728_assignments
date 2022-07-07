from numpy import result_type
from utils import load_config_file
from train import Trainer
import pandas as pd
from datasets import get_data

if __name__ == '__main__':

	ques = 2 # 1,2
	results_folder = '../results/'
	run_num = 3


	if ques == 1:
		cfg = load_config_file('./configs/elq_0.py')
		trainer = Trainer(cfg)
		max_acces = trainer.train()

	elif ques == 2:
		cfg_file = './configs/ati_0.py'
		cfg = load_config_file(cfg_file)
		trainer = Trainer(cfg)
		max_acces = trainer.train()
		print('Best Train Acc: '+str(max_acces['trn']))
		print('Best Test Acc: '+str(max_acces['test0']))
		trn_loss, _, _, trn_acc, trn_sent_labels = trainer.evaluate(1, 'trn', -1)
		test_loss, _, _, test_acc, test_sent_labels = trainer.evaluate(1, 'test', -1)

		num_to_coarse, coarse_to_num, num_to_fine, fine_to_num = get_data('', 4)
		coarse = 0
		if cfg.dataset.is_coarse:
			coarse = 1
			num_to_cls, cls_to_num = num_to_coarse, coarse_to_num
		else:
			num_to_cls, cls_to_num = num_to_fine, fine_to_num

		for i in range(len(trn_sent_labels)):
			trn_sent_labels[i][1] = num_to_cls[trn_sent_labels[i][1]]
			trn_sent_labels[i].append(trn_sent_labels[i][1].split(':')[0])
		for i in range(len(test_sent_labels)):
			test_sent_labels[i][1] = num_to_cls[test_sent_labels[i][1]]
			test_sent_labels[i].append(test_sent_labels[i][1].split(':')[0])

		result_file_trn = results_folder+str(ques)+'_trn_'+str(coarse)+'_'+str(run_num)+'.csv'
		result_file_test = results_folder+str(ques)+'_test_'+str(coarse)+'_'+str(run_num)+'.csv'

		df0 = pd.DataFrame(trn_sent_labels, index = None, columns = None)
		df1 = pd.DataFrame(test_sent_labels, index = None, columns = None)
		df0.to_csv(result_file_trn, index = False, header = False)
		df1.to_csv(result_file_test, index = False, header = False)

