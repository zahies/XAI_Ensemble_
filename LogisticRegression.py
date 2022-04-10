from sklearn.linear_model import LogisticRegression
from Comparison_Detection import RANDOM_SEED
from Preprocess_twitter import *
from Preprocess_German import *


def define_fit_Logistic(x_train, y_train, y_test, x_test, model_dataset_names, data_path):
    x_train, x_test, y_train, y_test = [x.to_numpy() for x in [x_train, x_test, y_train, y_test]]
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 0] = 0.5
    sample_weights[y_train == 1] = 4


    lr = LogisticRegression()

    if (os.path.isfile(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl") == False):
        lr.fit(x_train, y_train)
        joblib.dump(lr, data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    else:
        lr = joblib.load(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    y_pred = lr.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("LR Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)
    return lr

