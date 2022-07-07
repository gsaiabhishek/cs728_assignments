from utils import load_config_file
from train import Trainer
import pandas as pd

if __name__ == '__main__':

	cfg_file = 'configs/ncc.py'
	output_folder = '../output/'

	cfg = load_config_file(cfg_file)
	sigmoid_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
	# sigmoid_thresholds = [0.5]
	print('EMBEDDINGS: '+cfg.dataset.embed_type+', MERGE TYPE: '+cfg.dataset.merge_type)

	for i in sigmoid_thresholds:
		print('Sigmoid threshold: '+str(i))
		cfg.dataset.sigmoid_threshold = i
		trainer = Trainer(cfg)
		max_acces = trainer.train()
		loss0, corr0, tot0, acc0, words_and_scores0 = trainer.evaluate('test', i, 0)
		loss1, corr1, tot1, acc1, words_and_scores1 = trainer.evaluate('test', i, 1)
		print('r1: Best Acc: '+str(max_acces['test0']))
		print('r2: Best Acc: '+str(max_acces['test1']))

		output_file_0 = output_folder+cfg.dataset.embed_type+'_'+'r1_'+str(i)+'.csv'
		output_file_1 = output_folder+cfg.dataset.embed_type+'_''r2_'+str(i)+'.csv'

		df0 = pd.DataFrame(words_and_scores0, index = None, columns = None)
		df1 = pd.DataFrame(words_and_scores1, index = None, columns = None)
		df0.to_csv(output_file_0, index = False, header = False)
		df1.to_csv(output_file_1, index = False, header = False)



