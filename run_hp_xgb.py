import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import cross_validation
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials
import os

feat_folder = "./stage2_feat"
train_file = "./data/stage1_labels_all.csv"
test_file = "./data/stage2_sample_submission.csv"
out_folder = "./xgb_pred_stage2"
iter_num = 2000

trial_counter = 0

def run_pca(feat, pca_comp):
    pca = PCA(n_components = pca_comp)
    new_feat = pca.fit_transform(feat)
    print("INFO: ALL new feature has shape: " + str(new_feat.shape))
    print("INFO: PCA explains variance ratio: " + str(pca.explained_variance_ratio_))
    print("INFO: PCA totally explains variance: " + str(sum(pca.explained_variance_ratio_)))
    return new_feat

def params_gen(para_name):	
	#params = params_space[model_name]
	#TODO: make this better
	if para_name == "param_space_reg_xgb":
		params = param_space_reg_xgb
	return params
	
def xgb_obj(x_train, x_test, y_train, params):
	global trial_counter
        trial_counter += 1

	pca_comp = params['pca_comp']
	if pca_comp > 0:
		x_all_new = run_pca(np.concatenate((x_train,x_test),axis=0), pca_comp)
		x_train_new = x_all_new[0:x_train.shape[0],]
		x_test_new = x_all_new[x_train.shape[0]:,]
		print("INFO: x_train_new has shape: " + str(x_train_new.shape))
		print("INFO: x_test_new has shape: " + str(x_test_new.shape))
	else:
		x_train_new = x_train
		x_test_new = x_test 
	
	trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x_train_new, y_train, random_state=96, stratify=y_train, test_size=0.20)

	clf = xgb.XGBRegressor(max_depth=int(params['max_depth']),
                           n_estimators=int(params['n_estimators']),
                           min_child_weight=params['min_child_weight'],
                           learning_rate=params['learning_rate'],
                           nthread=8,
                           subsample=params['subsample'],
                           colsample_bytree=params['colsample_bytree'],
                           seed=params['seed'])

	clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
	best_evals_result = min(clf.evals_result()['validation_0']['logloss'])
	print("=======================")
	print("INFO: best_evals_result: %f" % best_evals_result)
	print("=======================")
	df = pd.read_csv(test_file)
	pred = clf.predict(x_test_new)

	df['cancer'] = pred
	if not os.path.exists(out_folder):
        	os.makedirs(out_folder)
	df.to_csv('%s/xgb_%s.csv' % (out_folder, str(trial_counter)), index=False)
	best_evals = {"value": [best_evals_result]}
	df_pred = pd.DataFrame(data = best_evals)
	df_pred.to_csv('%s/best_evals__%s.csv' % (out_folder, str(trial_counter)), index=False)	
	return {'loss': best_evals_result, 'status': STATUS_OK}	
	
	
def hyperopt_train_xgboost():
    df_train = pd.read_csv(train_file)
    df_test =  pd.read_csv(test_file)
    x_train = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_train['id'].tolist()])
    x_test = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_test['id'].tolist()])
    print("INFO: x_train has shape: " + str(x_train.shape))
    print("INFO: x_test has shape: " + str(x_test.shape))
    y_train = df_train['cancer'].as_matrix()
    name = "param_space_reg_xgb"
    params = params_gen(name)
    trials = Trials()
    obj = lambda p: xgb_obj(x_train, x_test, y_train, p)
    best = fmin(obj, params, tpe.suggest, iter_num, trials)
    fnvals = [t['result']['loss'] for t in trials.trials]
    best_log_loss = min(fnvals)
    print("*****************************************")
    print("Final Best Log-Loss: %.6f" % best_log_loss)
    trial_losses = np.asarray(trials.losses(), dtype=float)
    ind = np.where(trial_losses == best_log_loss)[0][0] + 1
    print("Final Best Log-Loss ind: %s" % str(ind))
    ind_data = {"value": [ind]}
    df_ind = pd.DataFrame(data = ind_data)
    df_ind.to_csv('%s/xgb_ind.csv' % out_folder, index=False)
    return best_log_loss, ind, best


param_space_reg_xgb = {
	'pca_comp': hp.choice('pca_comp', [-1, 30, 60, 90]),
	'max_depth': hp.quniform('max_depth', 3, 5, 1),
	'n_estimators': hp.quniform('n_estimators', 1000, 4000, 500),
	'min_child_weight' : hp.quniform('min_child_weight', 1.0, 100.0, 5.0),
	'learning_rate' : hp.quniform('learning_rate', 0.01, 0.05, 0.01),
	'subsample' : hp.quniform('subsample', 0.8, 0.9, 0.05),
	'colsample_bytree' : hp.quniform('colsample_bytree', 0.8, 0.9, 0.05),
	'seed' : 96
}

if __name__ == '__main__':
    best_log_loss, ind, best = hyperopt_train_xgboost()
    print("++++++++++++++++++++++++++++++")
    print(best_log_loss, ind, best)
	
