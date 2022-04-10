# from Preprocess_twitter import *
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


def define_fit_GBoost(x_train, y_train, y_test, x_test):
    x_train, x_test, y_train, y_test = [x.to_numpy() for x in [x_train, x_test, y_train, y_test]]
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 0] = 0.5
    sample_weights[y_train == 1] = 4


    # rf = RandomForestClassifier(**RF_PARAMS)
    # rf.fit(x_train, y_train)
    # gb = GradientBoostingClassifier(max_depth=12, n_estimators=250, random_state=RANDOM_SEED)
    gb = GradientBoostingClassifier(loss='exponential', learning_rate=0.05, n_estimators=250, random_state=RANDOM_SEED)
    # gb = AdaBoostClassifier(learning_rate=0.05, n_estimators=250, random_state=RANDOM_SEED)
    # xg = xgboost.XGBClassifier(objective="binary:logistic", max_depth=12, n_estimators=250, random_state=RANDOM_SEED)
    # rf = RandomForestClassifier(**RF_PARAMS)
    if (os.path.isfile("models/GBoost_twitter" + str(RANDOM_SEED) + ".pkl") == False):
        gb.fit(x_train, y_train, sample_weight=sample_weights)
        joblib.dump(gb, "models/GBoost_twitter" + str(RANDOM_SEED) + ".pkl")
    else:
        gb = joblib.load("models/GBoost_twitter" + str(RANDOM_SEED) + ".pkl")
    y_pred = gb.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("GBoost Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)
    return gb




