import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import cross_validation
import xgboost as xgb

feat_folder = "./out_feat"
pca_use = False
n_comp = 60

train_file = "./data/stage1_labels.csv"
test_file = "./data/stage1_sample_submission.csv"
out_folder = "./data"

def run_pca(feat):
    pca = PCA(n_components = n_comp)
    new_feat = pca.fit_transform(feat)
    print("INFO: ALL new feature has shape: " + str(new_feat.shape))
    print("INFO: PCA explains variance ratio: " + str(pca.explained_variance_ratio_))
    print("INFO: PCA totally explains variance: " + str(sum(pca.explained_variance_ratio_)))
    return new_feat

def train_xgboost():
    df_train = pd.read_csv(train_file)
    df_test =  pd.read_csv(test_file)
    x_train = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_train['id'].tolist()])
    x_test = np.array([np.mean(np.load('%s/%s.npy' % (feat_folder, str(id))), axis=0) for id in df_test['id'].tolist()])
    print("INFO: x_train has shape: " + str(x_train.shape))
    print("INFO: x_test has shape: " + str(x_test.shape))
    y_train = df_train['cancer'].as_matrix()
    if pca_use:
        x_all_new = run_pca(np.concatenate((x_train,x_test),axis=0))
        x_train_new = x_all_new[0:x_train.shape[0],]
        x_test_new = x_all_new[x_train.shape[0]:,]
        print("INFO: x_train_new has shape: " + str(x_train_new.shape))
        print("INFO: x_test_new has shape: " + str(x_test_new.shape))
    else:
        x_train_new = x_train
        x_test_new = x_test

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x_train_new, y_train, random_state=42, stratify=y_train,
                                                                   test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=5,
                           n_estimators=2500,
                           min_child_weight=96,
                           learning_rate=0.03737,
                           nthread=8,
                           subsample=0.85,
                           colsample_bytree=0.90,
                           seed=96)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
    return x_test_new, clf

def make_submit():
    x_test_new, clf = train_xgboost()
    df = pd.read_csv(test_file)
    pred = clf.predict(x_test_new)

    df['cancer'] = pred
    df.to_csv('%s/xgb.csv' % out_folder, index=False)
    print(df.head())


if __name__ == '__main__':
    make_submit()
