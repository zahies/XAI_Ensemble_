import numpy as np
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
from Preprocess_twitter import load_twitter_data
from Preprocess_KDD import load_kdd_data

# from Preprocess_German import load_german_data

from GradientBoost import define_fit_GBoost
from RandomForest import define_fit_RForestoost
from XGBoost import define_fit_XGBoost
from HateVersarial import Attack
from Comparison_Detection import RANDOM_SEED
from Preprocess_SDN import load_sdn_data
from Preprocess_IoT import load_iot_data
# from Preprocess_Lending import load_lending_data
# from Preprocess_Vehicle import load_vehicle_data
# from Preprocess_Beth import load_beth_data
# from Preprocess_Risk import load_risk_data
# from Preprocess_KDD import load_kdd_data

target_name = "XGBoost_HATE_"
# target_name = "XGBoost_"



# data_path = 'data_twitter'
# dataset_name = 'twitter_HATE'
# data_path = 'data_Lending'
# dataset_name = 'lending'
# data_path = 'data_NSL_KDD'
# dataset_name = 'kdd_test_HATE'
# data_path = 'data_beth'
# dataset_name = 'Beth_HATE'
# data_path = 'data_vehicle'
# dataset_name = 'vehicle_HATE'
# data_path = 'data_SDN'
# dataset_name = 'sdn_HATE'
# data_path = 'data_credit'
# dataset_name = 'RISK_HATE'
data_path = 'data_IoT'
dataset_name = 'iot_HATE'

# data_path = 'data_twitter_embd'
# dataset_name = 'twitter_embd'
# constraints:


class factory_cat():
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def integer(self, tensor):
        res = tf.round(tensor)
        return res

    def bound(self,tensor, t_min, t_max):
        res = tf.clip_by_value(tensor, t_min, t_max)
        return res

    def method(self, tensor):
        tensor = self.integer(tensor)
        return self.bound(tensor, self.min, self.max)

    def get_method(self):
        return self.method


tf_max_value = tf.constant(np.inf)
tf_min_value = tf.constant(-np.inf)


def bound(tensor, t_min, t_max):
    res = tf.clip_by_value(tensor, t_min, t_max)
    return res


def positive(tensor):
    res = bound(tensor, 0, tf_max_value)
    return res


def negative(tensor):
    res = bound(tensor,  tf_min_value, 0)
    return res



def integer(tensor):
    res = tf.round(tensor)
    return res


def categorical(tmin, tmax):
    return factory_cat(tmin, tmax).get_method()


def binary(tensor):
    tensor = integer(tensor)
    tensor = bound(tensor, 0, 1)
    return tensor


def normalized(tensor):
    return bound(tensor, 0, 1)



CONSTRAINTS = {
    'anger_empath':	[normalized],
    'appearance_empath':	[normalized],
    'attractive_empath':	[normalized],
    'banking_empath':	[positive],
    'betweenness':	[positive],
    'body_empath':	[normalized],
    'c_air_travel_empath':	[normalized],
    'c_art_empath':	[normalized],
    'c_banking_empath':	[normalized],
    'c_betweenness':	[positive],
    'c_childish_empath':	[normalized],
    'c_cleaning_empath':	[normalized],
    'c_computer_empath':	[normalized],
    'c_crime_empath':	[normalized],
    'c_dispute_empath':	[normalized],
    'c_divine_empath':	[normalized],
    'c_dominant_personality_empath':	[normalized],
    'c_eigenvector':	[positive],
    'c_exasperation_empath':	[normalized],
    'c_family_empath':	[normalized],
    'c_farming_empath':	[normalized],
    'c_fashion_empath':	[normalized],
    'c_fire_empath':	[normalized],
    'c_followees_count':	[positive, integer],
    'c_followers_count':	[positive, integer],
    'c_furniture_empath':	[normalized],
    'c_hate_empath':	[normalized],
    'c_hipster_empath':	[normalized],
    'c_hygiene_empath':	[normalized],
    'c_independence_empath':	[normalized],
    'c_irritability_empath':	[normalized],
    'c_joy_empath':	[normalized],
    'c_kill_empath':	[normalized],
    'c_legend_empath':	[normalized],
    'c_listen_empath':	[normalized],
    'c_lust_empath':	[normalized],
    'c_masculine_empath':	[normalized],
    'c_medical_emergency_empath':	[normalized],
    'c_medieval_empath':	[normalized],
    'c_money_empath':	[normalized],
    'c_number urls':	[positive],
    'c_optimism_empath':	[normalized],
    'c_out_degree':	[normalized],
    'c_power_empath':	[normalized],
    'c_rage_empath':	[normalized],
    'c_ridicule_empath':	[normalized],
    'c_shape_and_size_empath':	[normalized],
    'c_ship_empath':	[normalized],
    'c_sleep_empath':	[normalized],
    'c_social_media_empath':	[normalized],
    'c_superhero_empath':	[normalized],
    'c_swearing_terms_empath':	[normalized],
    'c_time_diff':	[positive],
    'c_tourism_empath':	[normalized],
    'c_traveling_empath':	[normalized],
    'c_trust_empath':	[normalized],
    'c_weakness_empath':	[normalized],
    'c_wedding_empath':	[normalized],
    'c_work_empath':	[normalized],
    'car_empath':	[positive],
    'cleaning_empath':	[normalized],
    'competing_empath':	[normalized],
    'created_at':	[positive, integer],
    'disgust_empath':	[normalized],
    'economics_empath':	[normalized],
    'fabric_empath':	[normalized],
    'favorites_count':	[positive, integer],
    'friends_empath':	[normalized],
    'gain_empath':	[normalized],
    'health_empath':	[normalized],
    'help_empath':	[normalized],
    'home_empath':	[normalized],
    'horror_empath':	[normalized],
    'irritability_empath':	[normalized],
    'joy_empath':	[normalized],
    'law_empath':	[normalized],
    'leader_empath':	[normalized],
    'liquid_empath':	[normalized],
    'lust_empath':	[normalized],
    'magic_empath':	[normalized],
    'morning_empath':	[normalized],
    'music_empath':	[normalized],
    'musical_empath':	[normalized],
    'nervousness_empath':	[normalized],
    'normal_neigh':	[binary],
    'hate_neigh':	[binary],
#     "is_50" : [binary],
#     "is_63" : [binary],
#     "is_50_2" : [binary],
#     "is_63_2" : [binary],
    'real_estate_empath':	[normalized],
    'sentiment':	[positive],
    'sexual_empath':	[normalized],
    'sleep_empath':	[normalized],
    'sound_empath':	[normalized],
    'sports_empath':	[normalized],
    'statuses_count':	[positive, integer],
    'surprise_empath':	[normalized],
    'tourism_empath':	[normalized],
    'urban_empath':	[normalized],
    'vacation_empath':	[normalized],
    'warmth_empath':	[normalized],
    'work_empath':	[normalized],
    'youth_empath':	[normalized],
    'zest_empath':	[normalized]
}

CATEGORICAL_FEATS = [col for col in CONSTRAINTS.keys() if binary in CONSTRAINTS[col] or isinstance(CONSTRAINTS[col][0], type(categorical(0,0)))]
NUMERIC_FEATS = [col for col in CONSTRAINTS.keys() if col not in CATEGORICAL_FEATS]

# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_twitter_data()
x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_iot_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_kdd_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_sdn_data()
# x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_risk_data()


# apply attack :
print("x_attack shape " , x_attack.shape)

half_set = int(len(x_train)/2)
x_train_target = x_train.iloc[half_set: len(x_train)]
y_train_target = y_train.iloc[half_set: len(y_train)]
""" different train set for training the substitute model of the attack (Black-Box Setting)"""
attack_train_x = x_train.iloc[:half_set]
attack_train_y = y_train[:half_set]


""" training the target model """
xg = define_fit_XGBoost(x_train_target, y_train_target, y_test, x_test, target_name + dataset_name, data_path)


""" apply attack """
attack = Attack(attack_train_x, attack_train_y, x_val, y_val, xg, CONSTRAINTS)
num_smpl_to_attack = 250
proto_critic_list, iter_num_list, df_adv = attack.cat_wrapper(x_attack.iloc[:num_smpl_to_attack], xg, 2, 2, attack_train_x)


print(df_adv.head())
print("df_adv shape:", df_adv.shape )
print("SUCCEEDED adv rate:", len(df_adv)/ num_smpl_to_attack, "%")
df_adv.to_csv(data_path + "/HATE_XG_adv_smpls_" + dataset_name + "_seed_"+str(RANDOM_SEED)+".csv")