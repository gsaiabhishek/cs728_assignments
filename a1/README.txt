The folder structure should be as follows:
180050036-a1
|-code/
|-data/ (submitted empty)
|	|-glove.6B/ (not submitting, please download and unzip this folder from https://nlp.stanford.edu/projects/glove/)
|	|-ncc/ (not submitting, given in moodle announcements)
|-output/ (has csv files containing words and predicted scores. File naming format: <glove/bert_pre_trained>_<r1/r2>_<threshold>.csv)
|-README.txt
|-results (has train and test loss for various epochs during training for glove and bert_pre_trained embeddings)
	|-glove_concat.txt
	|-bert_concat.txt
|-report.txt (has the best observed accuracy for r1 nad r2 for glove and bert_pre_trained embeddings)

Please install any libraries that are needed to run code. Ex: torch, transformers, dotmap
To run, change the config file ncc.py in code/configs/ as necessary and use the command `python3 run.py > ../results/output.txt` in 'code' directory

naming convention in results/:
tn_l = train loss
tn_a = train accuracy
similarly t_l, t_a for test loss, accuracy resp.

naming convention in report.txt:
merge_type is how we merged two words of the noun compound. Ex: concatenation

FFN dim for glove embed:
input_dim = 600,
hidden_dim_1 = 600,
hidden_dim_2 = 600,
output_dim = 1,

FFN dim for bert embed:
input_dim = 1536,
hidden_dim_1 = 1000,
hidden_dim_2 = 1000,
output_dim = 1,