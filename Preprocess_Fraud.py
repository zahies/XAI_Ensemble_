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
from imblearn.over_sampling import RandomOverSampler,SMOTE
from collections import Counter

TARGET = 'TARGET'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_fraud():

    if (os.path.isfile("data_Fraud/fraud_after_preprocess.csv") == False):
        to_fit = pd.read_csv("data_Fraud/application_data.csv", low_memory=False)
        # df = df.drop("Unnamed: 0", axis=1)
        # df.columns = df.columns.str.replace(' ', '_')
        print(to_fit.shape)

        print("Object type values:", np.count_nonzero(to_fit.select_dtypes('object').columns))
        print("___________________________________________________________________________________________")
        print(to_fit.select_dtypes('object').columns)
        print("___________________________________________________________________________________________")

        le = LabelEncoder()
        to_fit['NAME_CONTRACT_TYPE'] = le.fit_transform(to_fit['NAME_CONTRACT_TYPE'])
        to_fit['CODE_GENDER'] = le.fit_transform(to_fit['CODE_GENDER'])
        to_fit['FLAG_OWN_CAR'] = le.fit_transform(to_fit['FLAG_OWN_CAR'])
        to_fit['FLAG_OWN_REALTY'] = le.fit_transform(to_fit['FLAG_OWN_REALTY'])
        to_fit['NAME_TYPE_SUITE'] = le.fit_transform(to_fit['NAME_TYPE_SUITE'].astype(str))
        to_fit['NAME_INCOME_TYPE'] = le.fit_transform(to_fit['NAME_INCOME_TYPE'])
        to_fit['NAME_EDUCATION_TYPE'] = le.fit_transform(to_fit['NAME_EDUCATION_TYPE'])
        to_fit['NAME_FAMILY_STATUS'] = le.fit_transform(to_fit['NAME_FAMILY_STATUS'])
        to_fit['NAME_HOUSING_TYPE'] = le.fit_transform(to_fit['NAME_HOUSING_TYPE'])
        to_fit['OCCUPATION_TYPE'] = le.fit_transform(to_fit['OCCUPATION_TYPE'].astype(str))
        to_fit['WEEKDAY_APPR_PROCESS_START'] = le.fit_transform(to_fit['WEEKDAY_APPR_PROCESS_START'])
        to_fit['ORGANIZATION_TYPE'] = le.fit_transform(to_fit['ORGANIZATION_TYPE'])
        to_fit['FONDKAPREMONT_MODE'] = le.fit_transform(to_fit['FONDKAPREMONT_MODE'].astype(str))
        to_fit['HOUSETYPE_MODE'] = le.fit_transform(to_fit['HOUSETYPE_MODE'].astype(str))
        to_fit['WALLSMATERIAL_MODE'] = le.fit_transform(to_fit['WALLSMATERIAL_MODE'].astype(str))
        to_fit['EMERGENCYSTATE_MODE'] = le.fit_transform(to_fit['EMERGENCYSTATE_MODE'].astype(str))

        to_fit['FONDKAPREMONT_MODE'] = mode_impute(to_fit, 'FONDKAPREMONT_MODE')
        to_fit['WALLSMATERIAL_MODE'] = mode_impute(to_fit, 'WALLSMATERIAL_MODE')
        to_fit['HOUSETYPE_MODE'] = mode_impute(to_fit, 'HOUSETYPE_MODE')
        to_fit['EMERGENCYSTATE_MODE'] = mode_impute(to_fit, 'EMERGENCYSTATE_MODE')
        to_fit['OCCUPATION_TYPE'] = mode_impute(to_fit, 'OCCUPATION_TYPE')
        to_fit['NAME_TYPE_SUITE'] = mode_impute(to_fit, 'NAME_TYPE_SUITE')
        missing(to_fit.select_dtypes('object'))

        # to_fit = to_fit.select_dtypes('float').interpolate(method='linear', limit_direction='forward')
        # to_fit = to_fit.dropna(axis=1)

        to_fit = to_fit.interpolate(method='linear', limit_direction='forward')
        to_fit = to_fit.dropna(axis=1)
        missing(to_fit)

        print(to_fit.isnull().sum(axis=0))
        print(to_fit.info())

        zero_cols = []
        for col in to_fit.columns:
            sum = (to_fit[col] == 0).sum()
            print(col, "  ", sum)
            if sum > 100000:
                if col == 'TARGET':
                    continue
                zero_cols.append(col)

        print(to_fit.shape)
        to_fit.drop(zero_cols, axis=1, inplace=True)
        print(to_fit.shape)



        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['member_id','tot_cur_bal']] = scaler.fit_transform(to_fit[['member_id','tot_cur_bal']])
        #
        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['id','annual_inc','out_prncp_inv','total_pymnt','total_rec_prncp','total_rec_int']] = scaler.fit_transform\
        #     (to_fit[['id','annual_inc','out_prncp_inv','total_pymnt','total_rec_prncp','total_rec_int']])


        # to_fit["member_id"] = zscore(to_fit["member_id"])


        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['_Flow_Duration', 'Fwd_IAT_Total', '_Bwd_IAT_Max', '_Fwd_IAT_Min', '_Bwd_IAT_Min', '_Flow_IAT_Max']] = scaler\
        #     .fit_transform(to_fit[['_Flow_Duration', 'Fwd_IAT_Total', '_Bwd_IAT_Max', '_Fwd_IAT_Min', '_Bwd_IAT_Min', '_Flow_IAT_Max']])
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)


        # y = to_fit["Class"]
        # del to_fit["Class"]
        # to_fit = to_fit.replace(0,-1)
        # to_fit["Class"] = y


        # 64
        del to_fit["FLAG_MOBIL"]
        del to_fit["FLAG_CONT_MOBILE"]
        del to_fit["FLAG_OWN_REALTY"]
        del to_fit["REGION_RATING_CLIENT"]
        del to_fit["NAME_HOUSING_TYPE"]


        # 36
        # del to_fit["funded_amnt_inv"]

        # 25
        # del to_fit["acc_now_delinq"]


        # 16
        # del to_fit["installment"]



        zeros = (to_fit == 0).astype(int).sum(axis=1)
        drop_idx = []
        for idx, value in zeros.items():
            if value > 8:
                drop_idx.append(idx)

        to_fit = to_fit.drop(drop_idx)
        print(to_fit.shape)

        # int_rate = to_fit["int_rate"]
        # y = to_fit["grade"]
        # del to_fit["grade"]
        # scaler = preprocessing.MinMaxScaler(feature_range=(1,100))
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)
        # to_fit["grade"] = y
        # to_fit["int_rate"] = int_rate


        # int_rate = to_fit["int_rate"]
        # y = to_fit["grade"]
        # del to_fit["grade"]
        # to_fit = to_fit.apply(zscore)
        # to_fit["grade"] = y
        # to_fit["int_rate"] = int_rate

        print(to_fit.info())
        print_class_freq(to_fit)



        to_fit = to_fit.astype('float32')
        print(to_fit.info())
        to_fit.to_csv("data_Fraud/fraud_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_Fraud/fraud_after_preprocess.csv")
        to_fit = to_fit.drop("Unnamed: 0", axis=1)
        to_fit = to_fit.astype('float32')
    return to_fit



def colors(value):
    if value > 50 and value < 100:
        color = 'red'
    elif value > 154000 and value < 250000:
        color = 'red'
    elif value == 1 :
        color = 'blue'
    else:
        color = 'green'
    return 'color: %s' % color



def mode_impute(df,col):
    return df[col].fillna(df[col].mode()[0])



def missing(df):
    total = df.isnull().sum().sort_values(ascending = False)
    total = total[total>0]
    percent = df.isnull().sum().sort_values(ascending = False)/len(df)*100
    percent = percent[percent>0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percentage']).style.applymap(colors)



def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df[TARGET] == 1].shape[0])
    print("Class 0 frequency: ")
    print(df[df[TARGET] == 0].shape[0])


def balanced_train_data(df_train):
    """
    sampeling to generate balanced train dataset
    """


    #
    # fixed_rnd = 123
    # # test = df_train[df_train[TARGET] == 1]
    # attack_train_df = df_train[df_train[TARGET] == 1].sample(10000, axis=0, random_state=fixed_rnd)
    # df2 = df_train[df_train[TARGET] == 0].sample(len(attack_train_df), axis=0, random_state=fixed_rnd)
    # df_train = pd.concat([attack_train_df, df2],  axis=0, sort=False)
    # df_train = df_train.sample(frac=1, random_state=fixed_rnd).reset_index(drop=True)
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


def load_fraud_data():

    df = read_and_preprocess_fraud()
    # ################## Validation Strategy - 80/20 ##################
    balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(balanced_df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns


    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    print('before SMOTE:', Counter(y_train))
    sm = SMOTE(sampling_strategy='minority')
    X_train2, Y_train2 = sm.fit_resample(x_train, y_train)
    print('After SMOTE:', Counter(Y_train2))


    fixed_rnd = 123
    # test = df_train[df_train[TARGET] == 1]
    attack_train_df = df_train[df_train[TARGET] == 1].sample(10000, axis=0, random_state=fixed_rnd)
    df2 = df_train[df_train[TARGET] == 0].sample(len(attack_train_df), axis=0, random_state=fixed_rnd)
    df_train = pd.concat([attack_train_df, df2],  axis=0, sort=False)
    df_train = df_train.sample(frac=1, random_state=fixed_rnd).reset_index(drop=True)


    print("x_train.shape: ",x_train.shape)
    print("x_test.shape: ",x_test.shape)
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test


