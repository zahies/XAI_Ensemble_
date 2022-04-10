import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost
from sklearn.model_selection import train_test_split
from Comparison_Detection import RANDOM_SEED
import os
from sklearn import preprocessing
from scipy.stats import zscore


TARGET = 'label'
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_iot():

    if (os.path.isfile("data_IoT/IoT_after_preprocess.csv") == False):
        to_fit = pd.read_csv("data_IoT/BotNeTIoT-L01_label_NoDuplicates.csv")
        to_fit = to_fit.drop("Unnamed: 0", axis=1)
        print(to_fit.shape)
        print(to_fit.info())
        is_attack = to_fit.label.map(lambda a: 0 if a == 1 else 1)
        del to_fit['label']
        to_fit['label'] = is_attack
        print_class_freq(to_fit)
        del to_fit['HH_L0.1_covariance']
        del to_fit['HH_jit_L0.1_weight']
        del to_fit['HH_L0.1_mean']
        del to_fit['H_L0.1_variance']
        del to_fit['H_L0.1_mean']
        del to_fit['H_L0.1_weight']
        del to_fit['HpHp_L0.1_pcc']


        # to_fit["HH_jit_L0.1_mean"] = zscore(to_fit["HH_jit_L0.1_mean"])
        # to_fit["HH_L0.1_magnitude"] = zscore(to_fit["HH_L0.1_magnitude"])
        # to_fit["MI_dir_L0.1_mean"] = zscore(to_fit["MI_dir_L0.1_mean"])
        # to_fit["MI_dir_L0.1_mean"] = zscore(to_fit["MI_dir_L0.1_mean"])
        # to_fit["HpHp_L0.1_radius"] = zscore(to_fit["HpHp_L0.1_radius"])
        # to_fit["MI_dir_L0.1_variance"] = zscore(to_fit["MI_dir_L0.1_variance"])
        # to_fit["HH_L0.1_radius"] = zscore(to_fit["HH_L0.1_radius"])
        # to_fit["HpHp_L0.1_weight"] = zscore(to_fit["HpHp_L0.1_weight"])
        # to_fit["HpHp_L0.1_magnitude"] = zscore(to_fit["HpHp_L0.1_magnitude"])
        # to_fit["HH_L0.1_weight"] = zscore(to_fit["HH_L0.1_weight"])
        # to_fit["HpHp_L0.1_std"] = zscore(to_fit["HpHp_L0.1_std"])
        # to_fit["HH_L0.1_std"] = zscore(to_fit["HH_L0.1_std"])
        # to_fit["HpHp_L0.1_mean"] = zscore(to_fit["HpHp_L0.1_mean"])
        # to_fit["HH_L0.1_pcc"] = zscore(to_fit["HH_L0.1_pcc"])
        # to_fit["HpHp_L0.1_covariance"] = zscore(to_fit["HpHp_L0.1_covariance"])

        y = to_fit["label"]
        del to_fit["label"]
        to_fit = to_fit.apply(zscore)
        to_fit["label"] = y




        to_fit = to_fit.astype('float32')
        to_fit.to_csv("data_IoT/IoT_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_IoT/IoT_after_preprocess.csv")
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
    fixed_rnd = 876
    attack_train_df = df_train[df_train[TARGET] == 1].sample(5000, axis=0, random_state=fixed_rnd)
    df2 = df_train[df_train[TARGET] == 0].sample(len(attack_train_df), axis=0, random_state=fixed_rnd)
    df_train = pd.concat([attack_train_df, df2],  axis=0, sort=False)
    df_train = df_train.sample(frac=1, random_state=fixed_rnd).reset_index(drop=True)
    return df_train

def train_test_attack_split(df, num_of_adv):

    df_train, df_val= train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test= train_test_split(df_val, test_size=0.5, random_state=RANDOM_SEED)



    df_attack = df_train.sample(n=num_of_adv, random_state=RANDOM_SEED)
    df_train = df_train.drop(df_attack.index)
    df_attack = pd.concat([df_attack, df_test])
    # print("df_train.shape: ",df_train.shape)
    # print("df_test.shape: ",df_test.shape)

    return df_train, df_val, df_test, df_attack


def load_iot_data():

    df = read_and_preprocess_iot()
    # ################## Validation Strategy - 80/20 ##################
    balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(balanced_df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns
    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    print("x_train.shape: ",x_train.shape)
    print("x_test.shape: ",x_test.shape)
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test
