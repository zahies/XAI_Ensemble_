import pandas as pd
import numpy as np
import joblib
from sklearn import datasets


from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, MinMaxScaler
from art.attacks.evasion import ZooAttack, HopSkipJump, BoundaryAttack
# from art.attacks.evasion import ZooAttack, HopSkipJump, BoundaryAttack, ThresholdAttack, GeoDA, SimBA, LowProFool
from art.estimators.classification.scikitlearn import ScikitlearnGradientBoostingClassifier, ScikitlearnRandomForestClassifier
from Wrapper_XG import ScikitlearnXGBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from Comparison_Detection import *
import Wrapper_XG

# data_path = 'data_Lending'
# data_name = 'lending'
# data_path = 'data_Car_Ins'
# data_name = 'car_ins'
data_path = 'data_NSL_KDD'
data_name = 'kdd_test'
# data_path = 'data_SDN'
# data_name = 'sdn'
# data_path = 'data_credit'
# data_name = 'RISK'
# data_path = 'data_vehicle'
# data_name = 'vehicle'
# data_path = 'data_twitter'
# data_name = 'twitter'
# data_path = 'data_beth'
# data_name = 'Beth'
# data_path = 'data_IoT'
# data_name = 'iot'

# data_path = 'data_twitter_embd'
# data_name = 'twitter_embd'

class Attack:
    def __init__(self, x_attack, y, o_model, scaler=None):
        super().__init__()
        self.x_attack = x_attack
        self.o_model = o_model
        self.y = y


    # def generate_samples(self, n_adv=None):
    #     adv_samples = []
    #     for i in range(len(self.x_attack)):
    #         adv_sample = self.HopSkipJump_attack(self.x_attack.iloc[[i]], self.y.iloc[i],self.o_model)
    #         if adv_sample is not None:
    #             adv_samples.append(adv_sample)
    #             # return adv_samples
    #         # if i == 7:
    #         #     break
    #     print("adversarial_samples len: ", len(adv_samples))
    #     return adv_samples

    def binary_constraint(self, idx, sample):
        if (sample[0,idx] > 0.5):
            sample[0, idx] = 1
        else:
            sample[0, idx] = 0
        return sample

    def ZOO_attack(self, x_attack, y, tar_model):
        iter_step = 5
        x_adv = None
        num_succ = 0
        adversarial_samples = []

        xg_art = Wrapper_XG.ScikitlearnXGBoostClassifier(model=tar_model)
        # , variable_h=1
        # attack = ZooAttack(classifier=xg_art, targeted=False, max_iter=100, nb_parallel=16,
        #                    learning_rate=50, variable_h=8, use_resize=False, confidence=0.8)
        attack = ZooAttack(classifier=xg_art, targeted=False, max_iter=60, nb_parallel=15, learning_rate=5, variable_h=1, use_resize=False)
        for ex_idx in range(0, len(x_attack)):
            print("Idx: ", ex_idx)
            x_benign = x_attack[ex_idx:ex_idx + 1].copy().values
            y_benign = y[ex_idx:ex_idx + 1][:, np.newaxis].copy()
            print("************************** NEW SAMPLE *************************")

            for i in range(10):
                x_adv = attack.generate(x=x_benign, x_adv_init=x_adv, resume=True)
                print("Adversarial image at step %d." % (i * iter_step), "L2 error",
                      np.linalg.norm(np.reshape(x_adv[0] - x_benign, [-1])),
                      "and class label %d." % np.argmax(xg_art.predict(x_adv)[0]))

                attack.max_iter = iter_step

            """for twitter"""
            if data_name == 'twiiter':
                x_adv = self.binary_constraint(99, x_adv)
            """for german"""
            if data_name == 'german':
              for i in range(4,24):
                 x_adv = self.binary_constraint(i, x_adv)


            if np.argmax(xg_art.predict(x_adv)[0]) != np.argmax(xg_art.predict(x_benign)[0]):
                num_succ = num_succ + 1
                print("Attack Success!")
                adversarial_samples.append(x_adv[0].tolist())

            else:
                print("Attack Failure")
        print("adversarial_samples len: ", len(adversarial_samples))
        return adversarial_samples

    def Boundary_attack(self, x_attack, y, tar_model):
        iter_step = 5
        x_adv = None
        num_succ = 0
        adversarial_samples = []

        xg_art = Wrapper_XG.ScikitlearnXGBoostClassifier(model=tar_model)
        # , variable_h=1
        attack = BoundaryAttack(estimator=xg_art, targeted=False)
        for ex_idx in range(0, len(x_attack)):
            print("Idx: ", ex_idx)
            x_benign = x_attack[ex_idx:ex_idx + 1].copy().values
            y_benign = y[ex_idx:ex_idx + 1][:, np.newaxis].copy()
            print("************************** NEW SAMPLE *************************")

            for i in range(20):
                x_adv = attack.generate(x=x_benign, x_adv_init=x_adv, resume=True)
                print("Adversarial image at step %d." % (i * iter_step), "L2 error",
                      np.linalg.norm(np.reshape(x_adv[0] - x_benign, [-1])),
                      "and class label %d." % np.argmax(xg_art.predict(x_adv)[0]))

                attack.max_iter = iter_step

            """for twitter"""
            if data_name == 'twiiter':
                x_adv = self.binary_constraint(99, x_adv)
            """for german"""
            if data_name == 'german':
              for i in range(4,24):
                 x_adv = self.binary_constraint(i, x_adv)


            if np.argmax(xg_art.predict(x_adv)[0]) != np.argmax(xg_art.predict(x_benign)[0]):
                num_succ = num_succ + 1
                print("Attack Success!")
                adversarial_samples.append(x_adv[0].tolist())

            else:
                print("Attack Failure")
        print("adversarial_samples len: ", len(adversarial_samples))
        return adversarial_samples


    def Threshold_attack(self, x_attack, y, tar_model):
        iter_step = 5
        x_adv = None
        num_succ = 0
        adversarial_samples = []
        xg_art = Wrapper_XG.ScikitlearnXGBoostClassifier(model=tar_model)
        attack = LowProFool(classifier=xg_art)
        for ex_idx in range(0, len(x_attack)):
            print("Idx: ", ex_idx)
            x_benign = x_attack[ex_idx:ex_idx + 1].copy().values
            y_benign = y[ex_idx:ex_idx + 1][:, np.newaxis].copy()
            print("************************** NEW SAMPLE *************************")
            for i in range(10):
                x_adv = attack.generate(x=x_benign, x_adv_init=x_adv, resume=True)
                print("Adversarial image at step %d." % (i * iter_step), "L2 error",
                      np.linalg.norm(np.reshape(x_adv[0] - x_benign, [-1])),
                      "and class label %d." % np.argmax(xg_art.predict(x_adv)[0]))
                attack.max_iter = iter_step
            """for twitter"""
            if data_name == 'twiiter':
                x_adv = self.binary_constraint(99, x_adv)
            """for german"""
            if data_name == 'german':
              for i in range(4,24):
                 x_adv = self.binary_constraint(i, x_adv)
            if np.argmax(xg_art.predict(x_adv)[0]) != np.argmax(xg_art.predict(x_benign)[0]):
                num_succ = num_succ + 1
                print("Attack Success!")
                adversarial_samples.append(x_adv[0].tolist())
            else:
                print("Attack Failure")
        print("adversarial_samples len: ", len(adversarial_samples))
        return adversarial_samples


    def HopSkipJump_attack(self, x_attack, y, tar_model):
        iter_step = 5
        x_adv = None
        num_succ = 0
        adversarial_samples = []
        xg_art = Wrapper_XG.ScikitlearnXGBoostClassifier(model=tar_model)
        attack = HopSkipJump(classifier=xg_art, targeted=False, max_iter=0, max_eval=1000, init_eval=10)
        for ex_idx in range(0, len(x_attack)):
            print("Idx: ", ex_idx)
            x_benign = x_attack[ex_idx:ex_idx + 1].copy().values
            y_benign = y[ex_idx:ex_idx + 1][:, np.newaxis].copy()
            print("************************** NEW SAMPLE *************************")
            for i in range(10):
                x_adv = attack.generate(x=x_benign, x_adv_init=x_adv, resume=True)
                print("Adversarial image at step %d." % (i * iter_step), "L2 error",
                      np.linalg.norm(np.reshape(x_adv[0] - x_benign, [-1])),
                      "and class label %d." % np.argmax(xg_art.predict(x_adv)[0]))
                attack.max_iter = iter_step
            """for twitter"""
            if data_name == 'twiiter':
                x_adv = self.binary_constraint(99, x_adv)
            """for german"""
            if data_name == 'german':
              for i in range(4,24):
                 x_adv = self.binary_constraint(i, x_adv)
            if np.argmax(xg_art.predict(x_adv)[0]) != np.argmax(xg_art.predict(x_benign)[0]):
                num_succ = num_succ + 1
                print("Attack Success!")
                adversarial_samples.append(x_adv[0].tolist())
            else:
                print("Attack Failure")
        print("adversarial_samples len: ", len(adversarial_samples))
        return adversarial_samples

x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test= load_kdd_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_sdn_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_risk_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_twitter_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_iot_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_vehicle_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_beth_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test= load_car_data()

x_train, x_test, y_train, y_test = [x.to_numpy() for x in [x_train, x_test, y_train, y_test]]



xg = xgboost.XGBClassifier(objective="binary:logistic", max_depth=12, n_estimators=250, random_state=RANDOM_SEED)
# rf = RandomForestClassifier(**RF_PARAMS)
if (os.path.isfile(data_path + "/models/XGBoost_" + data_name + str(RANDOM_SEED) + ".pkl") == False):
    xg.fit(x_train, y_train)
    joblib.dump(xg, data_path + "/models/XGBoost_" + data_name + str(RANDOM_SEED) + ".pkl")
else:
    xg = joblib.load(data_path + "/models/XGBoost_" + data_name + str(RANDOM_SEED) + ".pkl")
y_pred = xg.predict(x_test)
acc = accuracy_score(y_test, y_pred)
# print("RandomForest Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)
print("XGBoost Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)


attack_hop = Attack(x_attack, y_attack, xg,scaler=None)

num_smpl_to_attack = 150
hop_samp = 0

# adversarial_samples = attack_hop.Threshold_attack(x_attack.iloc[:num_smpl_to_attack], y_attack, xg)
# adversarial_samples = attack_hop.HopSkipJump_attack(x_attack.iloc[hop_samp:num_smpl_to_attack], y_attack, xg)
# adversarial_samples = attack_hop.Boundary_attack(x_attack.iloc[:num_smpl_to_attack], y_attack, xg)
adversarial_samples = attack_hop.ZOO_attack(x_attack.iloc[:num_smpl_to_attack], y_attack, xg)
adversarial_samples = np.array(adversarial_samples)

# TODO: change path of out samples according to the attack type !

# attack_name = 'HopSkip'
# attack_name = 'Boundary'
attack_name = 'ZOO'



""" twitter"""
# df = read_and_preprocess_twitter()


""" german"""
# df_credit = pd.read_csv("data_twitter/german_credit_data.csv")

""" sdn """
# df = read_and_preprocess_sdn()

""" risk """
# df = read_and_preprocess_risk()

""" vehicle """
# df = read_and_preprocess_vehicle()

""" ioT """
# df = read_and_preprocess_iot()

""" beth """
# df = read_and_preprocess_beth()


""" kdd """

df = read_and_preprocess_kdd()

""" LendingClub """

# df = read_and_preprocess_lending()

""" Car Ins """

# df = read_and_preprocess_car()



df = df[0:0]
df = df.drop([TARGET],axis=1)

df_adv = pd.DataFrame(adversarial_samples)

"""EMBEDD"""
# df_adv.columns = df.columns


print("SUCCEEDED adv rate:", len(df_adv) / (num_smpl_to_attack - hop_samp) , "%")



df_adv.to_csv(data_path + "/" + attack_name + "_XG_adv_smpls_" + data_name + "_seed_"+str(RANDOM_SEED)+".csv")
# x_attack.iloc[:50].to_csv(data_path + "/attack_smpls_" + data_name + "_seed_"+str(RANDOM_SEED)+".csv")





print()

