import os

import joblib
import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
from Comparison_Detection import RANDOM_SEED
from tqdm import tqdm
from skimage.transform import resize
# from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from Preprocess_SDN import load_sdn_data



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


class Model():
    def __init__(self, x_train, y_train, y_test, x_test, model_dataset_names):
        xg = xgboost.XGBClassifier(objective="binary:logistic", max_depth=12, n_estimators=250,
                                   random_state=RANDOM_SEED)
        # rf = RandomForestClassifier(**RF_PARAMS)
        if (os.path.isfile("models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl") == False):
            xg.fit(x_train, y_train)
            joblib.dump(xg, "models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
        else:
            xg = joblib.load("models/" + model_dataset_names + str(RANDOM_SEED) + ".pkl")
        self.model = xg
        self.input_size = x_train.shape[1]
    def run_on_batch(self, x):
        return self.model.predict(x)


def generate_masks(N, s, p1, model):
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, model.input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size)
        y = np.random.randint(0, cell_size)
        # Linear upsampling and cropping
        masks[i, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x + model.input_size, y + model.input_size]
    masks = masks.reshape(-1, *model.input_size, 1)
    return masks



def explain(model, inp, masks):
    batch_size = 100
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    sal = sal / N / p1
    return sal

def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]



x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = load_sdn_data()
data_path = 'data_SDN'
dataset_name = 'sdn'

model = Model(x_train, y_train, x_test, y_test, "XGBoost_" + dataset_name)

N = 2000
s = 8
p1 = 0.5
masks = generate_masks(2000, 8, 0.5, model)
x = x_train.iloc[50, :]
sal = explain(model, x, masks)

print("DSAD")