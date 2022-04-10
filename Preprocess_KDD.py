import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost
from sklearn.model_selection import train_test_split
import os
from sklearn import preprocessing
from Comparison_Detection import RANDOM_SEED
from scipy.stats import zscore


TARGET = 'attack_flag'

n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_kdd():
    file_path_20_percent = 'data_NSL_KDD/KDDTrain+_20Percent.txt'
    file_path_full_training_set = 'data_NSL_KDD/KDDTrain+.txt'
    file_path_test = 'data_NSL_KDD/KDDTest+.txt'

    if (os.path.isfile("data_NSL_KDD/kdd_after_preprocess_test.csv") == False):
        df = pd.read_csv(file_path_20_percent)
        # df = pd.read_csv(file_path_full_training_set)
        # test_df = pd.read_csv(file_path_test)

        columns = (['duration'
            , 'protocol_type'
            , 'service'
            , 'flag'
            , 'src_bytes'
            , 'dst_bytes'
            , 'land'
            , 'wrong_fragment'
            , 'urgent'
            , 'hot'
            , 'num_failed_logins'
            , 'logged_in'
            , 'num_compromised'
            , 'root_shell'
            , 'su_attempted'
            , 'num_root'
            , 'num_file_creations'
            , 'num_shells'
            , 'num_access_files'
            , 'num_outbound_cmds'
            , 'is_host_login'
            , 'is_guest_login'
            , 'count'
            , 'srv_count'
            , 'serror_rate'
            , 'srv_serror_rate'
            , 'rerror_rate'
            , 'srv_rerror_rate'
            , 'same_srv_rate'
            , 'diff_srv_rate'
            , 'srv_diff_host_rate'
            , 'dst_host_count'
            , 'dst_host_srv_count'
            , 'dst_host_same_srv_rate'
            , 'dst_host_diff_srv_rate'
            , 'dst_host_same_src_port_rate'
            , 'dst_host_srv_diff_host_rate'
            , 'dst_host_serror_rate'
            , 'dst_host_srv_serror_rate'
            , 'dst_host_rerror_rate'
            , 'dst_host_srv_rerror_rate'
            , 'attack'
            , 'level'])

        df.columns = columns
        # test_df.columns = columns

        is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
        # test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

        # data_with_attack = df.join(is_attack, rsuffix='_flag')
        df['attack_flag'] = is_attack
        # test_df['attack_flag'] = test_attack

        # sanity check
        # df.head()


        print(df.shape)

        # print(df.info(verbose=True))
        print_class_freq(df)

        print(df.info())


        le = preprocessing.LabelEncoder()
        le.fit(df['protocol_type'])
        df['protocol_type'] = le.transform(df['protocol_type'])

        le = preprocessing.LabelEncoder()
        le.fit(df['service'])
        df['service'] = le.transform(df['service'])

        le = preprocessing.LabelEncoder()
        le.fit(df['flag'])
        df['flag'] = le.transform(df['flag'])

        # get the intial set of encoded features and encode them
        # features_to_encode = ['protocol_type', 'service', 'flag']
        # encoded = pd.get_dummies(df[features_to_encode])
        # test_encoded_base = pd.get_dummies(test_df[features_to_encode])

        # not all of the features are in the test set, so we need to account for diffs
        # test_index = np.arange(len(test_df.index))
        # column_diffs = list(set(encoded.columns.values) - set(test_encoded_base.columns.values))
        #
        # diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)
        #
        # # we'll also need to reorder the columns to match, so let's get those
        # column_order = encoded.columns.to_list()
        #
        # # append the new columns
        # test_encoded_temp = test_encoded_base.join(diff_df)
        #
        # # reorder the columns
        # test_final = test_encoded_temp[column_order].fillna(0)

        # get numeric features, we won't worry about encoding these at this point
        # numeric_features = ['duration', 'src_bytes', 'dst_bytes']
        #
        # # model to fit/test
        # to_fit = encoded.join(df[numeric_features])
        # test_set = test_final.join(test_df[numeric_features])
        to_fit = df
        to_fit['attack_flag'] = df['attack_flag']
        print(to_fit.head())



        # before test:
        # del to_fit['urgent']
        # del to_fit['land']
        # del to_fit['num_file_creations']
        # del to_fit['num_shells']
        # del to_fit['num_access_files']
        # del to_fit['num_outbound_cmds']
        # del to_fit["attack"]

        """testtt"""
        # y = to_fit["attack_flag"]
        # del to_fit["attack_flag"]
        # to_fit = to_fit.apply(zscore)
        # to_fit["attack_flag"] = y

        # scaler = preprocessing.MinMaxScaler()
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)

        # test:
        del to_fit["attack"]
        del to_fit['urgent']
        del to_fit['land']
        del to_fit['num_file_creations']
        del to_fit['num_shells']
        del to_fit['num_access_files']
        del to_fit['level']
        del to_fit['duration']
        del to_fit['num_compromised']
        del to_fit['srv_rerror_rate']
        del to_fit['wrong_fragment']
        del to_fit['num_root']
        del to_fit['is_guest_login']
        del to_fit['is_host_login']
        del to_fit['su_attempted']
        del to_fit['root_shell']
        del to_fit['num_failed_logins']
        del to_fit['num_outbound_cmds']


        to_fit = to_fit.astype('float32')
        print(to_fit.info())

        # least important features
        # del to_fit["service_login"]
        # del to_fit['service_nnsp']

        to_fit.to_csv("data_NSL_KDD/kdd_after_preprocess_test.csv")
    else:
        to_fit = pd.read_csv("data_NSL_KDD/kdd_after_preprocess_test.csv")
        to_fit = to_fit.drop("Unnamed: 0", axis=1)
        to_fit = to_fit.astype('float32')
    return to_fit

def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df[TARGET] == 1].shape[0])
    print("Class 0 frequency: ")
    print(df[df[TARGET] == 0].shape[0])

def balanced_train_data(df_train):
    """
    sampeling to generate balanced train dataset
    """
    hate_train_df = df_train[df_train.pred == 1]
    print("hate_train_df shape:", hate_train_df.shape)
    df2 = df_train[df_train[TARGET] == 0].sample(len(hate_train_df)*5, axis=0, random_state=RANDOM_SEED)
    df_train = pd.concat([hate_train_df, df2],  axis=0, sort=False)
    print("df shape: ", df_train.shape)
    return df_train

def train_test_attack_split(df, num_of_adv):

    df_train, df_val= train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test= train_test_split(df_val, test_size=0.5, random_state=RANDOM_SEED)



    df_attack = df_train.sample(n=num_of_adv, random_state=RANDOM_SEED)
    df_train = df_train.drop(df_attack.index)
    df_attack = pd.concat([df_attack, df_test])


    return df_train, df_val, df_test, df_attack


def load_kdd_data():

    df = read_and_preprocess_kdd()
    # ################## Validation Strategy - 80/20 ##################
    # balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns
    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    print("x_train.shape: ",x_train.shape)
    print("x_test.shape: ",x_test.shape)
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test


