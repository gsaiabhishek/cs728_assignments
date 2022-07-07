The folder structure should be as follows:
180050036-a3
|-code/
|    |-token/	(submitted empty, will contain tokens given by bert tokensier once we run run.py)
|    |-other understandable directories
|-README.txt (this file)
|-data/ (submitted empty)
|   |-train_5500.label.txt (for question 2, from https://cogcomp.seas.upenn.edu/Data/QA/QC/)
|   |-TREC_10.label.txt (for question 2, from https://cogcomp.seas.upenn.edu/Data/QA/QC/)
|   |-EL4QA_data/ (for question 1, from http://dl.fbaipublicfiles.com/elq/EL4QA_data.tar.gz)
|-output/ 	(has train and test loss for various epochs during training)
|			(naming convention: <ques_no>_<coarse>_<run_num>.txt)
|           (<coarse> is present only for 2nd question. It is 1 if trained with coarse labels. 0 if trained with fine labels)
|-results/ 	(has csv files containing sentences and predicted labels. File naming format:  <ques_no>_<trn/test>_<coarse>_<run_num>.csv)
|           (<coarse> is present only for 2nd question. It is 1 if trained with coarse labels. 0 if trained with fine labels)
|           (for question 2: csv files has 3 columns: Senetence | label(fine or coarse) | label(coarse). 2nd column is based
|           on what granularity of labels the data is trained with.)
|-report.txt (has the best observed accuracy and other explanations)

Please install any libraries that are needed to run code. Ex: torch, transformers(**>=4.17.0**), dotmap, sklearn
To run, change the config file ati_0.py (for question 2) in code/configs/ as necessary and use the command `python3 run.py > ../output/1.txt` in 'code' directory
Internet is needed to run transformers(**>=4.17.0**) package based bert

For any explanations, please see report.txt.

There is also code from assignment 1 and 2 in the same files. Ex: NCDataset class in datasets/dataset.py,
configs/ncc.py, configs/fine_ncc_<num>.py etc...Please Ignore them.

naming convention in output/:
tn_l = train loss
tn_a = train accuracy
similarly t_l0, t_a0 for test loss, accuracy resp.
