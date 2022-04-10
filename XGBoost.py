# from Preprocess_twitter import *
# from Preprocess_KDD import RANDOM_SEED
import os

import joblib
import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
from Comparison_Detection import RANDOM_SEED
import pickle

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


def define_fit_XGBoost(x_train, y_train, y_test, x_test, model_dataset_names, data_path):
    x_train, x_test, y_train, y_test = [x.to_numpy() for x in [x_train, x_test, y_train, y_test]]
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 0] = 0.5
    sample_weights[y_train == 1] = 4


    # rf = RandomForestClassifier(**RF_PARAMS)
    # rf.fit(x_train, y_train)

    xg = xgboost.XGBClassifier(objective="binary:logistic", max_depth=12, n_estimators=250, random_state=RANDOM_SEED)
    # rf = RandomForestClassifier(**RF_PARAMS)
    if (os.path.isfile(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl") == False):
        print(" Training XGB...")
        xg.fit(x_train, y_train)
        joblib.dump(xg, data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    else:
        xg = joblib.load(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
        # pickle.dump(xg, open(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl", "wb"))
        # xg = pickle.load(open(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl", "rb"))
    y_pred = xg.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("XGBoost Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)
    return xg
