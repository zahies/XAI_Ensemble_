from Preprocess_twitter import *
from Preprocess_German import *
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def define_fit_RForestoost(x_train, y_train, y_test, x_test, model_dataset_names, data_path):
    x_train, x_test, y_train, y_test = [x.to_numpy() for x in [x_train, x_test, y_train, y_test]]
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 0] = 0.5
    sample_weights[y_train == 1] = 4

    RF_PARAMS = {
        'n_estimators': 250,
        'bootstrap': True,
        'criterion': 'gini',
        'random_state': RANDOM_SEED,
        'max_depth': 12,
        'max_leaf_nodes': 91,
        #     'min_impurity_split': 0.05,
    }

    rf = RandomForestClassifier(**RF_PARAMS)

    if (os.path.isfile(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl") == False):
        rf.fit(x_train, y_train)
        joblib.dump(rf, data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    else:
        rf = joblib.load(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    y_pred = rf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("RandomForest Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)
    return rf

