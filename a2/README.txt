The folder structure should be as follows:
180050036-a2
|-code/
|-README.txt (this file)
|-embed/	(submitted empty since the 'saved' bert embeddings in .pkl file is too large to submit. One can run
|			the code to genereate them anyway)
|-data/ (submitted empty)
|	|-ncc/ 	(not submitting, given in google doc. Link:
|	|		https://iitbacin.sharepoint.com/:u:/s/iitb_2022_cs728/EVbyaPrga-ZEqgkebc2X3poBuzNDxyMy22oo44icc1bPww?e=Kzr2hN)
|	|-glove.6B	(not submitting, please download and unzip this folder from https://nlp.stanford.edu/projects/glove/)
|	|-train.json
|	|-test.json
|			(train and test jsons download from here: 
|			https://drive.google.com/drive/folders/1lM3Mm1oYhC_OMh8ve9M6_VSQSbJYN2Fy?usp=sharing)
|
|-output/ 	(has train and test loss for various epochs during training for bert_pre_trained embeddings)
|			(naming convention: <1/0>_<1/0>_<1/0>_<run_num>.txt. Where 1/0 corresponds to indiviudal word bert
|			embed, compound word bert embed, contextual sentences bert embed used or not respectively. all 0's mean glove embed)
|			(IMP: 0 accuracies in test data for 0_0_1 ie, when contextual sentences embed are used is because we don't have gold labels there)
|-results/ 	(has csv files containing words and predicted labels. File naming format:  <1/0>_<1/0>_<1/0>_<trn/test>_<run_number>.csv. 
|			Where 1/0 corresponds to indiviudal word bert embed, compound word bert embed, contextual sentences bert embed used
|			 or not respectively?)
|-report.txt (has the best observed accuracy for th dataset, contextual using bert_pre_trained embeddings AND PART B)

Please install any libraries that are needed to run code. Ex: torch, transformers(**>=4.17.0**), dotmap, sklearn
To run, change the config file ncc.py in code/configs/ as necessary and use the command `python3 run.py > ../output/1.txt` in 'code' directory
Internet is needed to run transformers(**>=4.17.0**) package based bert

For subjective PART B, please see report.txt.

There is also code from assignment 1 in the same files. Ex: NCDataset class in datasets/dataset.py,
configs/ncc.py etc...Please Ignore them.

naming convention in output/:
tn_l = train loss
tn_a = train accuracy
similarly t_l0, t_a0 for test loss, accuracy resp.

FFN dim | expmnt_type in run.py:

	only glove embed individual words are used | 0: 
	input_dim = 300*2,
	hidden_dim_1 = 480,
	hidden_dim_2 = 480,
	output_dim = 37,

	only bert embed from individual words are used | 1: 
	input_dim = 768*2,
	hidden_dim_1 = 1200,
	hidden_dim_2 = 1200,
	output_dim = 37,

	both bert embed from individual words and compund word are used | 2: 
	input_dim = 768*4,
	hidden_dim_1 = 2400,
	hidden_dim_2 = 2400,
	output_dim = 37,

	only bert embed from contextual sentences are used | 3: 
	input_dim = 768*2,
	hidden_dim_1 = 1200,
	hidden_dim_2 = 1200,
	output_dim = 37,

training for the files in output/ : no of epochs
	Run 1:
	0_0_0_1 : 25
	1_0_0_1 : 25
	1_1_0_1 : 25
	0_0_1_1 : 25

	Run 2:
	1_0_0_2 : 100
	1_1_0_2 : 100
	0_0_1_2 : 100