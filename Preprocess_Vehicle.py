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



TARGET = 'Response'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_vehicle():

    if (os.path.isfile("data_vehicle/vehicle_insurance_after_preprocess.csv") == False):
        train = pd.read_csv("data_vehicle/vehicle_insurance.csv")

        print(train.shape)

        num_feat = ['Age', 'Vintage']
        cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year',
                    'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel']

        train['Gender'] = train['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        train = pd.get_dummies(train, drop_first=True)
        train = train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                                      "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
        train['Vehicle_Age_lt_1_Year'] = train['Vehicle_Age_lt_1_Year'].astype('int')
        train['Vehicle_Age_gt_2_Years'] = train['Vehicle_Age_gt_2_Years'].astype('int')
        train['Vehicle_Damage_Yes'] = train['Vehicle_Damage_Yes'].astype('int')

        ss = preprocessing.StandardScaler()
        train[num_feat] = ss.fit_transform(train[num_feat])

        mm = preprocessing.MinMaxScaler()
        train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

        train = train.drop('id', axis=1)
        for column in cat_feat:
            train[column] = train[column].astype('str')

        print_class_freq(train)
        to_fit = train




        # del to_fit["Vehicle_Age_lt_1_Year"]
        # del to_fit["Driving_License"]

        # test
        del to_fit["Vehicle_Damage_Yes"]
        del to_fit["Driving_License"]


        to_fit = to_fit.astype('float32')
        print(to_fit.info())
        to_fit.to_csv("data_vehicle/vehicle_insurance_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_vehicle/vehicle_insurance_after_preprocess.csv")
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
    fixed_rnd = 321
    attack_train_df = df_train[df_train[TARGET] == 1].sample(2500, axis=0, random_state=fixed_rnd)
    print("attack_train_df shape:", attack_train_df.shape)
    df2 = df_train[df_train[TARGET] == 0].sample(len(attack_train_df)*2, axis=0, random_state=fixed_rnd)
    df_train = pd.concat([attack_train_df, df2],  axis=0, sort=False)
    df_train = df_train.sample(frac=1, random_state=fixed_rnd).reset_index(drop=True)
    print("df shape: ", df_train.shape)
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


def load_vehicle_data():

    df = read_and_preprocess_vehicle()
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

