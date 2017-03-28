import numpy as np
import pandas as pd
from param_config import new_config
from core.models.hyperopt_tune import model_hyperopt_tune
from sklearn.cross_validation import StratifiedKFold
from core.data_proc import Data_proc

def prepCV(config, dfTrain):	
	skf = [0]*config.n_run
	for run in range(config.n_run):
		seed = 2017 + 1000 * (run + 1)
		# each run, use different seed to split train/valid data
		skf[run] = StratifiedKFold(dfTrain, n_folds=config.n_fold, shuffle=True, random_state=seed)
		for fold, (trainInd, validInd) in enumerate(skf[run]):
			print("=================================")
			print("TrainInd for run: %d fold: %d" % (run+1, fold+1))
			print(trainInd[:5])
			print("ValidInd for run: %d fold: %d" % (run+1, fold+1))
			print(validInd[:5])
	print("INFO: CV prepared and stored successfully")
	return skf



if __name__ == '__main__':
	#1. Generate training, test dataset and save them 
	df_train = pd.read_csv(train_file)
	df_test =  pd.read_csv(test_file)
	dfTrainX = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_train['id'].tolist()])
	dfTestX = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_test['id'].tolist()])
	print("INFO: dfTrainX has shape: " + str(dfTrainX.shape))
	print("INFO: dfTestX has shape: " + str(dfTestX.shape))
	dfTrainY = df_train['cancer'].as_matrix()
	files = (dfTrainX, dfTrainY, dfTestX)
	filenames = ("dfTrainX", "dfTrainY", "dfTestX")
	Data_proc.persist(config, files, filenames)
	
	#2. Generate stratifiedKFold_ind
	skf = prepCV(config, dfTrainX)
	Data_proc.persist(config, (skf,), ("stratifiedKFold_ind",))
	
	
	#3. Set feature and models, run them
	feat_name = "all"
	model_list = ["param_space_reg_xgb_linear", "param_space_reg_xgb_tree", "param_space_cls_skl_rf"]
	
	for model_name in model_list:
		best = model_hyperopt_tune(new_config, feat_name, model_name, loss_fct=None)
		print("################%s################/n%s" % (model_name, best))