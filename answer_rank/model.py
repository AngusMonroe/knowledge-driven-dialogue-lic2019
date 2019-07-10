# Xgb ranker

import numpy as np
from xgboostextension import XGBRanker
from pkl_util import load_pkl, to_pkl
import config
import json
import codecs
from glob import glob



def train(train_data, y, num_each_group):
    print("Start Training...")
    CASE_NUM = train_data.shape[0]
    GROUPS_NUM = int(CASE_NUM/num_each_group)
    assert CASE_NUM % GROUPS_NUM ==0

    X_groups = np.arange(0, GROUPS_NUM).repeat(num_each_group)

    X = np.concatenate([X_groups[:, None], train_data], axis=1)

    ranker = XGBRanker(n_estimators=150, learning_rate=0.1, subsample=1.0,max_depth=6)

    ranker.fit(X, y, eval_metric=['ndcg', 'map@5-'])

    to_pkl(ranker, config.model_save_path)
    return ranker


def predict(test_data, ranker, num_each_group, predict_save):
    print("Start predicting...")
    fw = codecs.open(predict_save, 'w')
    CASE_NUM = test_data.shape[0]
    GROUPS_NUM = int(CASE_NUM / num_each_group)
    assert CASE_NUM % GROUPS_NUM == 0

    X_groups = np.arange(0, GROUPS_NUM).repeat(num_each_group)
    X = np.concatenate([X_groups[:, None], test_data], axis=1)
    y_pred = ranker.predict(X)
    y_pred = y_pred.reshape(-1, 3)
    res = y_pred.argmax(axis=1).tolist()
    to_pkl(res, config.predict_index_save)
    with open(config.test_sample_file, 'r') as fr:
        for ix, line in zip(res, fr):
            data = json.loads(line.strip("\n"))
            preds = data["preds"]
            pred = preds[ix]
            fw.write(pred+"\n")
    fw.close()


if __name__ == '__main__':
    if len(glob(config.model_save_path)) == 0:
        train_data = load_pkl(config.train_file)
        y = load_pkl(config.target_file)
        print(y[:12])
        ranker = train(train_data, y, 3)
    else:
        ranker = load_pkl(config.model_save_path)
    test_data = load_pkl(config.test_file)
    predict(test_data, ranker, 3, config.predict_save)

