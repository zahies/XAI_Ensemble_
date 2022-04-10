import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost
from sklearn.model_selection import train_test_split
import os




TARGET = 'Risk_bad'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_german():
    df_credit = pd.read_csv("data_german/german_credit_data.csv", index_col=0)
    print(df_credit.shape)

    # print(df.info(verbose=True))
    print_class_freq(df_credit)

    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)


    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

    # Purpose to Dummies Variable
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True,
                                right_index=True)
    # Sex feature in dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True,
                                right_index=True)
    # Housing get dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True,
                                right_index=True)
    # Housing get Saving Accounts
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'),
                                left_index=True, right_index=True)
    # Housing get Risk
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
    # Housing get Checking Account
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'),
                                left_index=True, right_index=True)
    # Housing get Age categorical
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'),
                                left_index=True, right_index=True)

    # Excluding the missing columns
    del df_credit["Saving accounts"]
    del df_credit["Checking account"]
    del df_credit["Purpose"]
    del df_credit["Sex"]
    del df_credit["Housing"]
    del df_credit["Age_cat"]
    del df_credit["Risk"]
    del df_credit['Risk_good']
    print(df_credit.info())
    df_credit = df_credit.astype('float32')
    df_credit.to_csv("data_german/german_after_preprocess.csv")
    return df_credit

def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df.Risk == 'bad'].shape[0])
    print("Class 0 frequency: ")
    print(df[df.Risk == 'good'].shape[0])

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
    print("df_train.shape: ",df_train.shape)
    print("df_test.shape: ",df_test.shape)

    return df_train, df_val, df_test, df_attack


def load_german_data():

    df = read_and_preprocess_german()
    # ################## Validation Strategy - 80/20 ##################
    # balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns
    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test


