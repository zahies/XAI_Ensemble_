from sklearn import svm
import os
from Comparison_Detection import RANDOM_SEED
import joblib
from sklearn.metrics import accuracy_score
from keras.models import Model
import numpy as np




def define_fit_SVM(x_train, y_train, y_test, x_test, model_dataset_names, data_path):
    x_train, x_test, y_train, y_test = [x.to_numpy() for x in [x_train, x_test, y_train, y_test]]
    if (os.path.isfile(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl") == False):
        rbf_svc = svm.SVC(kernel='rbf', probability=True)
        rbf_svc.fit(x_train, y_train)
        joblib.dump(rbf_svc, data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    else:
        rbf_svc = joblib.load(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")

    y_pred = rbf_svc.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("SVM Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)



    y_pred_prob = rbf_svc.predict_proba(x_train)
    x_test_svm_2 = rbf_svc.predict_proba(x_test)

    rbf_svc_2 = svm.SVC(kernel='rbf', probability=True)
    rbf_svc_2.fit(y_pred_prob, y_train)
    y_pred = rbf_svc_2.predict(x_test_svm_2)
    acc = accuracy_score(y_test, y_pred)
    print("SVM_2 Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)


    return rbf_svc, rbf_svc_2


def define_fit_SVM_extract(x_train, y_train, y_test, x_test, model_dataset_names, data_path, CNN):
        print(CNN.summary())
        extract_1 = Model(inputs=CNN.inputs, outputs=CNN.layers[-4].output)
        extract_2 = Model(inputs=CNN.inputs, outputs=CNN.layers[-6].output)
        extract_3 = Model(inputs=CNN.inputs, outputs=CNN.layers[-8].output)

        if (os.path.isfile(data_path + "/models/_1" + model_dataset_names + str(RANDOM_SEED) + ".pkl") == False):
            embd_1 = extract_1.predict(x_train)
            rbf_svc_1 = svm.SVC(kernel='rbf', probability=True)
            rbf_svc_1.fit(embd_1, y_train)
            svc_prob_1 = rbf_svc_1.predict_proba(embd_1)

            embd_2 = extract_2.predict(x_train)
            rbf_svc_2 = svm.SVC(kernel='rbf', probability=True)
            rbf_svc_2.fit(embd_2, y_train)
            svc_prob_2 = rbf_svc_2.predict_proba(embd_2)

            joblib.dump(rbf_svc_1, data_path + "/models/_1" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
            joblib.dump(rbf_svc_2, data_path + "/models/_2" + model_dataset_names + str(RANDOM_SEED) + ".pkl")

        # embd_3 = extract_3.predict(x_train)
        # rbf_svc_3 = svm.SVC(kernel='rbf', probability=True)
        # rbf_svc_3.fit(embd_3, y_train)
        # svc_prob_3 = rbf_svc_3.predict_proba(embd_3)

            conc_features = np.concatenate((svc_prob_1, svc_prob_2), axis=1)
            rbf_svc_4 = svm.SVC(kernel='rbf', probability=True)
            rbf_svc_4.fit(conc_features, y_train)
            joblib.dump(rbf_svc_4, data_path + "/models/_3" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
        print("DONE fitting SVMs")


def pred_svm(test, model_dataset_names, data_path, CNN):
    extract_1 = Model(inputs=CNN.inputs, outputs=CNN.layers[-4].output)
    extract_2 = Model(inputs=CNN.inputs, outputs=CNN.layers[-6].output)

    embd_1 = extract_1.predict(test)
    rbf_svc_1 = joblib.load(data_path + "/models/_1" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    pred_1 = rbf_svc_1.predict_proba(embd_1)


    embd_2 = extract_2.predict(test)
    rbf_svc_2 = joblib.load(data_path + "/models/_2" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    pred_2 = rbf_svc_2.predict_proba(embd_2)

    rbf_svc_3 = joblib.load(data_path + "/models/_3" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
    conc_features_test = np.concatenate((pred_1, pred_2), axis=1)
    preds = rbf_svc_3.predict_proba(conc_features_test)
    return preds