import numpy as np
import shap
RANDOM_SEED = 123456789
# from RandomForest import *
# from LogisticRegression import *
from XGBoost import *
from CNN_tabular import *
from AE_SVM import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
# from Preprocess_twitter import *
# from Preprocess_Risk import *
from Preprocess_IoT import *
# from Preprocess_KDD import *
# from DNR_tabular import *
import lime
import lime.lime_tabular
from Preprocess_German import load_german_data
# from Preprocess_SDN import *
# from Preprocess_Vehicle import *
# from Preprocess_Ddos import *
# from Preprocess_Beth import *
# from Preprocess_spam import *
# from Preprocess_Fraud import *
# from Preprocess_Lending import load_lending_data
# from Preprocess_CarIns import load_car_data





# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    data_path = 'data_IoT'
    dataset_name = 'iot'
    # data_path = 'data_credit'
    # dataset_name = 'RISK'
    # data_path = 'data_twitter'
    # dataset_name = 'twitter'
    # data_path = 'data_NSL_KDD'
    # dataset_name = 'kdd_test'
    # data_path = 'data_SDN'
    # dataset_name = 'sdn'
    # data_path = 'data_vehicle'
    # dataset_name = 'vehicle'
    # data_path = 'data_Ddos'
    # dataset_name = 'Ddos'
    # data_path = 'data_beth'
    # dataset_name = 'Beth'
    # data_path = 'data_Fraud'
    # dataset_name = 'fraud'
    # data_path = 'data_spam'
    # dataset_name = 'spam'
    # data_path = 'data_Lending'
    # dataset_name = 'lending_NO_SCALE'
    # data_path = 'data_Car_Ins'
    # dataset_name = 'car_ins'

    # data_path = 'data_twitter_embd'
    # dataset_name = 'twitter_embd'

    # attack_name = 'HATE'
    attack_name = 'HopSkip'
    # attack_name = 'ZOO'
    # attack_name = 'Boundary'



    """ Three THRESHOLDS """
    # min SHAP difference of benign - 0.000..1
    THRESHOLD_ADV = 0.23429914655033356

    # max prediction difference of benign + 0.000..1
    THRESHOLD_DIST = 0.002

    # min Lime-SHAP difference of benign - 0.000..1
    THRESHOLD_ADV_LIME = -0.97380526869502


    """ loading dataset """
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_lending_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_car_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_fraud_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_sdn_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_vehicle_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_ddos_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test, embed_model= load_twitter_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_spam_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_beth_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_german_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_risk_data()
    x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_iot_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_twitter_data()
    # x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_kdd_data()




    """ half set for the model or not (half set for black-box settings) """
    # if the attack is HATE choose the names with HATE
    cnn_name = "CNN_"
    target_name = "XGBoost_"

    """ black-box setting for HateVersarial attack """
    if attack_name == 'HATE':
        half_set = int(len(x_train) / 2)
        x_train = x_train.iloc[half_set: len(x_train)]
        y_train = y_train.iloc[half_set: len(y_train)]
        cnn_name = "CNN_HATE_"
        target_name = "XGBoost_HATE_"
        dataset_name = dataset_name + "_HATE"


    print("----------------", dataset_name, "----------------")
    shap_test = x_train[0:50]
    x_train.iloc[50, :] = make_unique_row(x_train.iloc[50, :],x_train.shape[1])
    x_train.iloc[51, :] = make_unique_row(x_train.iloc[51, :],x_train.shape[1])
    x_train.iloc[52, :] = make_unique_row(x_train.iloc[52, :],x_train.shape[1])
    x_train.iloc[53, :] = make_unique_row(x_train.iloc[53, :],x_train.shape[1])
    x_train.iloc[54, :] = make_unique_row(x_train.iloc[54, :],x_train.shape[1])
    x_train.iloc[58, :] = make_unique_row(x_train.iloc[58, :],x_train.shape[1])
    x_train.iloc[57, :] = make_unique_row(x_train.iloc[57, :],x_train.shape[1])
    # x_train.iloc[56, :] = make_unique_row(x_train.iloc[56, :],x_train.shape[1])
    # x_train.iloc[55, :] = make_unique_row(x_train.iloc[55, :],x_train.shape[1])
    # x_train.iloc[59, :] = make_unique_row(x_train.iloc[59, :],x_train.shape[1])
    # x_train.iloc[60, :] = make_unique_row(x_train.iloc[60, :],x_train.shape[1])
    # x_train.iloc[61, :] = make_unique_row(x_train.iloc[61, :],x_train.shape[1])
    # x_train.iloc[62, :] = make_unique_row(x_train.iloc[62, :],x_train.shape[1])

    # for col in x_train.columns:
    #     print(col)
    #     # print(x_train[col].value_counts())
    #     print(x_train[col].describe())

    # model = load_model(data_path + "/models/" + "CNN_" + dataset_name + str(RANDOM_SEED) + ".h5")
    # print(model.summary())


    x_train_gb = x_train[50: len(x_train)]
    # x_train = x_train[20: len(x_train)]
    y_train_gb = y_train[50:len(y_train)]






    """ generating shap values from XGBoost """
    gb = define_fit_XGBoost(x_train_gb, y_train_gb, y_test, x_test, target_name + dataset_name, data_path)
    gb_explainer = shap.TreeExplainer(gb)
    shap_values_gb = gb_explainer.shap_values(shap_test)
    # shap.summary_plot(shap_values_gb, shap_test, plot_type="bar", max_display=20, title=dataset_name, plot_size=(15,15))

    """ sorted features importance names"""
    vals = np.abs(shap_values_gb).mean(0)
    feature_importance = pd.DataFrame(list(zip(x_train.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.head()



    """ compare only most important features """
    # indices_important = []
    # num_important = 5
    # for i in range(num_important):
    #     indices_important.append(x_train.columns.get_loc(feature_importance['col_name'].iloc[i]))
    #
    # shap_most_xg = []
    # for k in range(len(shap_values_gb)):
    #     most_xg = []
    #     for i in range(num_important):
    #         most_xg.append(shap_values_gb[k, indices_important[i]])
    #     shap_most_xg.append(most_xg)
    # shap_most_xg = np.array(shap_most_xg)


    """ lime """
    # sorted_lime = lime_exp.local_exp[1]

    # save and load limes
    if os.path.isfile(data_path + "/arranged_limes_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl") == False:
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train_gb.to_numpy(), feature_names=x_train_gb.columns,
                                                           class_names=['0', '1'], verbose=True)
        arranged_limes = []
        lime_idx = 0
        for x in shap_test.iterrows():
            print("Index : ", lime_idx)
            lime_idx += 1
            lime_exp = explainer.explain_instance(x[1].to_numpy(), gb.predict_proba, num_features=9)
            arranged_lime = [0] * x_train_gb.shape[1]
            for tup in lime_exp.local_exp[1]:
                arranged_lime[tup[0]] = tup[1]
            arranged_limes.append(arranged_lime)
        arranged_limes = np.array(arranged_limes)
        joblib.dump(arranged_limes,
                    data_path + "/arranged_limes_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
    else:
        arranged_limes = joblib.load(data_path + "/arranged_limes_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")

    """ generating shap values from Dense """
    # reg_dnn = getDenseModel('kdd')
    # model = fit_model_dnn(reg_dnn, x_train_gb, y_train_gb, x_test, y_test, x_val, y_val, 'CNN_kdd')
    # x_train_gb = x_train_gb.to_numpy()
    # background = x_train_gb[np.random.choice(x_train_gb.shape[0], 3000, replace=False)]
    # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    # e = shap.DeepExplainer(model, background)
    # shap_test = shap_test.to_numpy()
    # shap_values_reg_dnn = e.shap_values(shap_test)


    """ generating shap values from RF """
    # rf = define_fit_RForestoost(x_train_gb, y_train_gb, y_test, x_test, "Rforest_" + dataset_name, data_path)
    # rf_explainer = shap.TreeExplainer(rf)
    # shap_values_rf = rf_explainer.shap_values(shap_test)[1]
    #
    # print("shap values size: ", len(shap_values_gb))
    # print("shap values size: ", shap_values_gb.shape)

    """ generating shap values from LR """
    # lr = define_fit_Logistic(x_train_gb, y_train_gb, y_test, x_test, "Logistic_" + dataset_name, data_path)
    # lr_explainer = shap.explainers.Linear(lr, masker=shap.maskers.Impute(data=x_train_gb))
    # shap_values_lr = lr_explainer.shap_values(shap_test)
    #
    # print("shap values size: ", len(shap_values_gb))
    # print("shap values size: ", shap_values_gb.shape)

    """  generating shap values from DNN """

    # images_train =  transform_data_to_naive_img(x_train)
    # images_val =  transform_data_to_naive_img(x_val)
    # images_test =  transform_data_to_naive_img(x_test)

    if os.path.isfile(data_path + "/tab_img_idx_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl") == False:
        images_train = transform_data_to_img(x_train, y_train)
        images_test_zahi = images_train[0:50]
        images_train = images_train[50:len(images_train)]
        tab_img_idx = find_uniq_idx(images_train, x_train[50:len(x_train)], dataset_name)
        len_idx = len(tab_img_idx)
        testtttttt = transform_new_img_to_exist_idx(x_train.iloc[50:51], tab_img_idx, dataset_name)

        testtttttt = np.repeat(testtttttt[..., np.newaxis], 1, -1)
        ll = []
        ll.append(testtttttt)
        flatten_test = flatten_shap_images(ll)
        arranged_test = arrange_idx(flatten_test, tab_img_idx, dataset_name)

        images_val = transform_new_img_to_exist_idx(x_val, tab_img_idx, dataset_name)
        images_test = transform_new_img_to_exist_idx(x_test, tab_img_idx, dataset_name)
        # images_val = transform_data_to_img(x_val, y_val)
        # images_test = transform_data_to_img(x_test, y_test)
        joblib.dump(tab_img_idx, data_path + "/tab_img_idx_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        joblib.dump(images_train, data_path + "/images_train_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        joblib.dump(images_val, data_path + "/images_val_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        joblib.dump(images_test, data_path + "/images_test_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        joblib.dump(images_test_zahi, data_path + "/images_test_zahi_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
    else:
        tab_img_idx = joblib.load(data_path + "/tab_img_idx_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        images_train = joblib.load(data_path + "/images_train_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        # images_train.reshape(20097, 9, 9)
        images_val = joblib.load(data_path + "/images_val_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        images_test = joblib.load(data_path + "/images_test_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")
        images_test_zahi = joblib.load(data_path + "/images_test_zahi_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")

    # plot_image(images_test_zahi,0, "Benign")

    # tab_img_idx = find_uniq_idx(images_test, x_test)
    # tab_img_idx = find_idx(images_train[:1], x_train[:1])
    # tab_img_idx2 = find_idx(images_train[1:2], x_train[1:2])
    # tab_img_idx3 = find_idx(images_train[2:3], x_train[2:3])

    model = getPreTrainedModel(dataset_name)
    print("train shape: ", images_train.shape)
    images_train = np.repeat(images_train[..., np.newaxis], 1, -1)
    images_val = np.repeat(images_val[..., np.newaxis], 1, -1)
    images_test = np.repeat(images_test[..., np.newaxis], 1, -1)
    images_test_zahi = np.repeat(images_test_zahi[..., np.newaxis], 1, -1)
    print("train shape: ", images_train.shape)
    model = fit_model(model, images_train, y_train[50:len(y_train)], images_test, y_test, images_val, y_val, cnn_name + dataset_name, data_path)

    # print(model.summary())


    # select a set of background examples to take an expectation over
    background = images_train
    if dataset_name == 'kdd' or  dataset_name == 'car_ins' or dataset_name == 'kdd_test' or dataset_name == 'sdn'\
            or dataset_name == 'iot' or dataset_name == 'kdd_test_HATE':
        # background = images_train[np.random.choice(images_train.shape[0], 3000, replace=False)]
        # background = images_train.sample(3000, axis=0, random_state=RANDOM_SEED)
        background = images_train[:3000, :]
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    """explain predictions of the model on images"""
    e = shap.DeepExplainer(model, background)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)

    shap_values_dnn = e.shap_values(images_test_zahi)
    flatten_shap_images_dnn = flatten_shap_images(shap_values_dnn)
    arranged_shap_dnn_values = arrange_idx(flatten_shap_images_dnn, tab_img_idx, dataset_name)
    arrrnged_np_test = np.array(arranged_shap_dnn_values)
    testt1 = np.array(arranged_shap_dnn_values[0])
    testt2 = np.array(arranged_shap_dnn_values[1])
    testt3 = np.array(arranged_shap_dnn_values[10])
    # plot the feature attributions
    # shap.image_plot(shap_values, -images_test[1:5])

    """ compare only most important features """

    # shap_most_cnn = []
    # for k in range(len(arrrnged_np_test)):
    #     most_cnn = []
    #     for i in range(num_important):
    #         most_cnn.append(arrrnged_np_test[k, indices_important[i]])
    #     shap_most_cnn.append(most_cnn)
    # shap_most_cnn = np.array(shap_most_cnn)


    # shap.summary_plot(arrrnged_np_test, shap_test, plot_type="bar", max_display=20, title=dataset_name, plot_size=(15, 15))
    print("fdsf")

    """ NEW EXP """
    # cols = x_train.columns
    # df_reg = pd.DataFrame(data=shap_values_gb, columns=cols)
    # df_dnn = pd.DataFrame(data=arranged_shap_dnn_values, columns=cols)
    # df_dnn = df_dnn.add_suffix('_dnn')
    # concat_features = []
    # mergerd_df = pd.concat(df_reg, df_dnn)
    # for i in range(len(images_test_zahi)):
    #     conc = np.concatenate((arranged_shap_dnn_values[i], shap_values_gb[i]), axis=0)
        # print("result ", i, " : ", manhattan)
        # if cosine >= THRESHOLD_ADV:
        # if manhattan <= THRESHOLD_ADV:
        #     sum_above_90 += 1

    """  comparing shap values """

    # benign samples

    # scaler = preprocessing.MinMaxScaler()
    # scaled_shap_values_gb = scaler.fit_transform(shap_values_gb)
    # scaled_shap_values_CNN = scaler.fit_transform(arrrnged_np_test)



    def manhattan_distance(x, y):
        return sum(abs(a - b) for a, b in zip(x, y))



    # """ DNR """
    # # svm, svm_2 = define_fit_SVM(x_train_gb, y_train_gb, y_test, x_test, "SVM_" + dataset_name, data_path)
    # define_fit_SVM_extract(images_train, y_train[50:len(y_train)], y_test, images_test, "SVM_" + dataset_name, data_path, model)
    # prob_svm = pred_svm(images_test_zahi, "SVM_" + dataset_name, data_path, model)
    # # print("Mean class 0 : ", np.mean(prob_svm[:,0]))
    # # print("Mean class 1 : ", np.mean(prob_svm[:,1]))
    # # argmax = np.argmax(prob_svm_third, axis=1)
    # max = np.max(prob_svm, axis=1)
    # print("Mean highest pred score : ", np.mean(max))

    # target_pred = gb.predict(shap_test)
    # auxiliary_pred = model.predict_classes(images_test_zahi).reshape(-1)
    target_pred = gb.predict_proba(shap_test)
    target_pred = target_pred[:,1]
    target_classes = np.rint(target_pred)
    auxiliary_pred = model.predict_proba(images_test_zahi).reshape(-1)
    auxiliary_classes = np.rint(auxiliary_pred)



    # print("------------Experiment Seed: ", RANDOM_SEED, "------------------")
    # print("---------------benign---------------")
    #
    # sum_benign = 0
    # disagreement = 0
    # sum_dist = 0
    # max_dist = 0
    # min_shap = 1
    # min_lime = 1
    # for i in range(len(images_test_zahi)):
    #     check_not_benign = False
    #     # manhattan = manhattan_distance(arranged_shap_dnn_values[i], shap_values_gb[i])
    #     # euqlidean = np.linalg.norm(arranged_shap_dnn_values[i] - shap_values_gb[i])
    #     # cosine = 1 - spatial.distance.cosine(shap_most_cnn[i], shap_most_xg[i])
    #     cosine = 1 - spatial.distance.cosine(arranged_shap_dnn_values[i], shap_values_gb[i])
    #     # cosine = 1 - spatial.distance.cosine(shap_values_rf[i], shap_values_gb[i])
    #     # cosine = 1 - spatial.distance.cosine(shap_values_lr[i], shap_values_gb[i])
    #     lime_shap = 1 - spatial.distance.cosine(arranged_limes[i], shap_values_gb[i])
    #     abs_dist = np.absolute(target_pred[i] - auxiliary_pred[i])
    #     print("Cosine ", i, " : ", cosine, " Target Pred: ", target_pred[i], " CNN Pred: " ,auxiliary_pred[i], " Abs Distance: ",abs_dist)
    #     print("Lime - Shap Similarity: ", lime_shap)
    #
    #
    #     # SHAP & Pred
    #     # if abs_dist <= THRESHOLD_DIST or cosine > THRESHOLD_ADV:
    #     #     sum_benign += 1
    #
    #
    #     # 1.Pred 2. SHAP
    #     # if abs_dist <= THRESHOLD_DIST and cosine > THRESHOLD_ADV:
    #     #     sum_benign += 1
    #
    #
    #     # LIME
    #     # if lime_shap >= THRESHOLD_ADV_LIME:
    #     #     sum_benign += 1
    #
    #
    #     # Pred
    #     # if abs_dist <= THRESHOLD_DIST:
    #     #     sum_benign += 1
    #
    #
    #     # 1. SHAP 2. Pred 3. LIME
    #     if abs_dist <= THRESHOLD_DIST and cosine > THRESHOLD_ADV and lime_shap > THRESHOLD_ADV_LIME:
    #         sum_benign += 1
    #
    #
    #     # 1.Pred 2.SHAP 3. LIME
    #     # if abs_dist <= THRESHOLD_DIST and cosine > THRESHOLD_ADV and lime_shap > THRESHOLD_ADV_LIME:
    #     #     sum_benign += 1
    #
    #
    #     # 1. SHAP 2. Pred
    #     # if abs_dist <= THRESHOLD_DIST and cosine > THRESHOLD_ADV:
    #     #     sum_benign += 1
    #
    #     # 1. SHAP 2. Pred
    #     # if cosine >= THRESHOLD_ADV:
    #     #     sum_above_90 += 1
    #     #     check_not_benign = True
    #     # if check_not_benign == True:
    #     #     if abs_dist > THRESHOLD_DIST:
    #     #         sum_above_90 -= 1
    #
    #     # SHAP
    #     # if cosine >= THRESHOLD_ADV:
    #     #     sum_benign += 1
    #
    #     if lime_shap < min_lime:
    #         min_lime = lime_shap
    #     if cosine < min_shap:
    #         min_shap = cosine
    #     if abs_dist > max_dist:
    #         max_dist = abs_dist
    #     if target_classes[i] != auxiliary_classes[i]:
    #         disagreement += 1
    #     sum_dist += abs_dist
    # print("---------Results-----------")
    # print(" Rate benign ", THRESHOLD_ADV, " similarity : ", sum_benign / len(images_test_zahi))
    # print(" Disagreement rate ", disagreement / len(images_test_zahi))
    # print(" Average Prediction Distance : ", sum_dist / len(images_test_zahi))
    # print(" Max Prediction Distance : ", max_dist)
    # print(" Min Cosine Score : ", min_shap)
    # print(" Min Lime-SHAP Score : ", min_lime)

    """ preprocess for adversarial samples """
    # adv samples
    adv_smpls_df = pd.read_csv(data_path + "/" + attack_name + "_XG_adv_smpls_" + dataset_name + "_seed_" + str(RANDOM_SEED) + ".csv")
    adv_smpls_df = adv_smpls_df.drop("Unnamed: 0", axis=1)
    # print(adv_smpls.info(verbose=True))
    print("adv_smpls shape: ", adv_smpls_df.shape)
    adv_smpls = adv_smpls_df.to_numpy()
    """ embedd """
    # extract = Model(inputs=embed_model.inputs, outputs=embed_model.layers[-3].output)
    # embed_adv = extract.predict(adv_smpls_df)
    # embed_adv = pd.DataFrame(data=embed_adv)


    adv_imgs = transform_new_img_to_exist_idx(adv_smpls_df, tab_img_idx, dataset_name)
    # adv_imgs = transform_new_img_to_exist_idx(embed_adv, tab_img_idx, dataset_name)

    # plot_image(adv_imgs, 0, "Adversarial - same sample")
    adv_imgs = np.repeat(adv_imgs[..., np.newaxis], 1, -1)

    # """ DNR """
    # prob_svm_adv = pred_svm(adv_imgs, "SVM_" + dataset_name, data_path, model)
    # max_adv = np.max(prob_svm_adv, axis=1)
    # print("Mean highest pred score : ", np.mean(max_adv))

    # adv shap with GB
    # adv_shap_gb = gb_explainer.shap_values(embed_adv.to_numpy())
    adv_shap_gb = gb_explainer.shap_values(adv_smpls)

    """Attack Agnostic"""
    shap_values_gb_train = gb_explainer.shap_values(x_train_gb)
    shap_values_gb_test = gb_explainer.shap_values(x_test)

    ae, svm = AE_SVM(shap_values_gb_train, shap_values_gb_test, data_path, dataset_name)

    AE_SVM_test(shap_values_gb_test, ae, svm, "Benign")
    AE_SVM_test(adv_shap_gb, ae, svm, "Adv")


    """ compare only most important features """

    # shap_most_xg_adv = []
    # for k in range(len(adv_shap_gb)):
    #     most_xg_adv = []
    #     for i in range(num_important):
    #         most_xg_adv.append(adv_shap_gb[k, indices_important[i]])
    #     shap_most_xg_adv.append(most_xg_adv)
    # shap_most_xg_adv = np.array(shap_most_xg_adv)

    # adv shap with RF
    # adv_shap_rf = rf_explainer.shap_values(embed_adv.to_numpy())[1]

    # adv shap with LR
    # adv_shap_lr = lr_explainer.shap_values(adv_smpls)

    """ generating adv shap values from Dense """
    # shap_values_reg_dnn_adv = e.shap_values(adv_smpls)



    if (os.path.isfile(data_path + "/" + attack_name + "_XG_adv_smpls_" + dataset_name + "_seed_" + str(RANDOM_SEED) + ".pkl") == False):
        # adv shap with CNN
        adv_shap_dnn = e.shap_values(adv_imgs)
        adv_flatten_dnn = flatten_shap_images(adv_shap_dnn)
        adv_arg_shap_dnn = arrange_idx(adv_flatten_dnn, tab_img_idx, dataset_name)
        joblib.dump(adv_arg_shap_dnn, data_path + "/" + attack_name + "_XG_adv_smpls_" + dataset_name + "_seed_" + str(RANDOM_SEED) + ".pkl")
    else:
        adv_arg_shap_dnn = joblib.load(data_path + "/" + attack_name + "_XG_adv_smpls_" + dataset_name + "_seed_" + str(RANDOM_SEED) + ".pkl")
    testt2_adv = np.array(adv_arg_shap_dnn[0])

    """ compare only most important features """
    adv_arg_shap_dnn = np.array(adv_arg_shap_dnn)

    # shap_most_cnn_adv = []
    # for k in range(len(adv_shap_gb)):
    #     most_cnn_adv = []
    #     for i in range(num_important):
    #         most_cnn_adv.append(adv_arg_shap_dnn[k, indices_important[i]])
    #     shap_most_cnn_adv.append(most_cnn_adv)
    # shap_most_cnn_adv = np.array(shap_most_cnn_adv)



    # scaler = preprocessing.MinMaxScaler()
    # scaled_shap_values_gb_adv = scaler.fit_transform(adv_shap_gb)
    # scaled_shap_values_CNN_adv = scaler.fit_transform(np.array(adv_arg_shap_dnn))





    # save and load adv. limes
    if os.path.isfile(data_path + "/" + attack_name + "arranged_limes_ADV_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl") == False:

        """ lime adv """
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train_gb.to_numpy(), feature_names=x_train_gb.columns,
                                                           class_names=['0', '1'], verbose=True)
        arranged_limes_adv = []
        # for x in adv_smpls_df.iloc[:50].iterrows():
        lime_idx_adv = 0
        for x in adv_smpls_df.iterrows():
            print("Index : ", lime_idx_adv)
            lime_idx_adv += 1
            lime_exp = explainer.explain_instance(x[1].to_numpy(), gb.predict_proba, num_features=9)
            arranged_lime = [0] * x_train_gb.shape[1]
            for tup in lime_exp.local_exp[1]:
                arranged_lime[tup[0]] = tup[1]
            arranged_limes_adv.append(arranged_lime)
        arranged_limes_adv = np.array(arranged_limes_adv)

        joblib.dump(arranged_limes_adv,
                    data_path + "/" + attack_name + "arranged_limes_ADV_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")

    else:
        arranged_limes_adv = joblib.load(data_path + "/" + attack_name + "arranged_limes_ADV_seed_" + dataset_name + "_" + str(RANDOM_SEED) + ".pkl")


    # target_pred_adv = gb.predict(adv_smpls)
    # auxiliary_pred_adv = model.predict_classes(adv_imgs).reshape(-1)


    target_pred_adv = gb.predict_proba(adv_smpls)
    target_pred_adv = target_pred_adv[:,1]
    target_classes_adv = np.rint(target_pred_adv)
    auxiliary_pred_adv = model.predict_proba(adv_imgs).reshape(-1)
    auxiliary_classes_adv = np.rint(auxiliary_pred_adv)



    # print("---------------adversarial---------------")
    # sum_adv = 0
    # disagreement_adv = 0
    # sum_dist_adv = 0
    # max_dist_adv = 0
    # for i in range(len(adv_smpls_df)):
    #     is_adv = False
    #     # cosine =  1 - spatial.distance.cosine(adv_arg_shap_dnn[i], adv_shap_gb[i])
    #     # manhattan = manhattan_distance(adv_arg_shap_dnn[i], adv_shap_gb[i])
    #     # euqlidean =  np.linalg.norm(adv_arg_shap_dnn[i] - adv_shap_gb[i])
    #     # cosine = 1 - spatial.distance.cosine(shap_most_cnn_adv[i], shap_most_xg_adv[i])
    #     cosine = 1 - spatial.distance.cosine(adv_arg_shap_dnn[i], adv_shap_gb[i])
    #     # cosine = 1 - spatial.distance.cosine(adv_shap_rf[i], adv_shap_gb[i])
    #     # cosine = 1 - spatial.distance.cosine(adv_shap_lr[i], adv_shap_gb[i])
    #     lime_shap_adv = 1 - spatial.distance.cosine(arranged_limes_adv[i], adv_shap_gb[i])
    #     abs_dist_adv = np.absolute(target_pred_adv[i] - auxiliary_pred_adv[i])
    #     print("Cosine adv", i, " : ", cosine, " Target Pred: ", target_pred_adv[i], " CNN Pred: ", auxiliary_pred_adv[i])
    #     print("Lime-SHAP Similarity: ", lime_shap_adv)
    #
    #
    #
    #     # SHAP & Pred
    #     # if abs_dist_adv > THRESHOLD_DIST and cosine <= THRESHOLD_ADV:
    #     #     sum_adv += 1
    #
    #
    #     # 1.Pred 2. SHAP
    #     # if abs_dist_adv > THRESHOLD_DIST or cosine <= THRESHOLD_ADV:
    #     #     sum_adv += 1
    #
    #
    #     # Pred
    #     # if abs_dist_adv > THRESHOLD_DIST:
    #     #     sum_adv += 1
    #
    #     # LIME
    #     # if lime_shap_adv < THRESHOLD_ADV_LIME:
    #     #     sum_adv += 1
    #
    #     # 1.Shap 2. Pred 3. LIME
    #     if abs_dist_adv > THRESHOLD_DIST or cosine <= THRESHOLD_ADV or lime_shap_adv <= THRESHOLD_ADV_LIME:
    #         sum_adv += 1
    #
    #     # 1.Shap 2. Pred
    #     # if abs_dist_adv > THRESHOLD_DIST or cosine <= THRESHOLD_ADV:
    #     #     sum_adv += 1
    #
    #     # 1.Pred 2.SHAP 3. LIME
    #     # if abs_dist_adv > THRESHOLD_DIST or cosine <= THRESHOLD_ADV or lime_shap_adv <= THRESHOLD_ADV_LIME:
    #     #     sum_adv += 1
    #
    #     # SHAP
    #     # if cosine < THRESHOLD_ADV:
    #     #     sum_adv += 1
    #
    #
    #     # 1.Shap 2. Pred
    #     """ check if adversary """
    #     # if cosine < THRESHOLD_ADV:
    #     #     sum_below_90 += 1
    #     #     is_adv = True
    #     # if is_adv == False:
    #     #     if abs_dist_adv > THRESHOLD_DIST:
    #     #         sum_below_90 += 1
    #
    #     if abs_dist_adv > max_dist_adv:
    #         max_dist_adv = abs_dist_adv
    #     if target_classes_adv[i] != auxiliary_classes_adv[i]:
    #         disagreement_adv += 1
    #     sum_dist_adv += abs_dist_adv
    # print("---------Results-----------")
    # print(" Detection rate : ", sum_adv / len(adv_smpls_df))
    # print("-----------------------------")
    # print(" Disagreement rate : ", disagreement_adv / len(adv_smpls_df))
    # print(" Average Prediction Distance Adv. : ", sum_dist_adv / len(adv_smpls_df))
    # print(" Max Prediction Distance : ", max_dist_adv)


    # print(" detection rate : ", sum_below_90 / len(arranged_limes_adv))

    # dnnn.predict(adv_imgs[0:10])
    print()
