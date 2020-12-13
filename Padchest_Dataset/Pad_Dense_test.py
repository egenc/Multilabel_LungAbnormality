#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import glob, time, os, cv2, csv

try:
    get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
    get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')
except NameError:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from tqdm import tqdm
import shutil
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, optimizers, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import ZeroPadding2D, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Activation, Flatten, Dropout, SpatialDropout2D, MaxPooling2D, GlobalMaxPooling2D, Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K

#from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


df_path = '/home/Erdal.Genc/covid_work/PadChest_csv/Padchest_filtered_withPath.csv'
img_path = '/mnt/dsets/ChestXrays/PadChest/image_zips'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())  ## to use which GPU in use
data_copy=pd.read_csv(df_path, low_memory=False)

img_height=224 #299
img_width=224 #299

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

# In[3]:


import tensorflow as tf
##Loading the model


model = load_model('/home/Erdal.Genc/covid_work/Padchest_work/')


# In[ ]:





# In[4]:


datagen=ImageDataGenerator(rescale=1./255., validation_split = 0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=data_copy,
directory="/mnt/dsets/ChestXrays/PadChest/image_zips",
x_col="Path",
y_col='new_labels',
subset="training",
batch_size=32,
seed=42,
shuffle=True,
classes = ['fibrosis', 'pleural_mass', 'pleural_thickening', 'mass', 'fissure', 
           'pneumothorax', 'costophrenic_angle_blunting', 'atelectasis', 'granuloma', 
           'other_lung_abnormality', 'interstitial_pattern', 'emphysema', 'consolidation', 
           'pleural_effusion', 'pulmonary_edema', 'ground_glass_pattern', 
           'infiltration', 'nodule', 'calcification', 'normal', 'pneumonia'],
class_mode="categorical",
target_size=(224,224))

valid_generator =datagen.flow_from_dataframe(
dataframe=data_copy,
directory="/mnt/dsets/ChestXrays/PadChest/image_zips",
x_col="Path",
y_col='new_labels',
subset="validation",
batch_size=32,
classes = ['fibrosis', 'pleural_mass', 'pleural_thickening', 'mass', 'fissure', 
           'pneumothorax', 'costophrenic_angle_blunting', 'atelectasis', 'granuloma', 
           'other_lung_abnormality', 'interstitial_pattern', 'emphysema', 'consolidation', 
           'pleural_effusion', 'pulmonary_edema', 'ground_glass_pattern', 
           'infiltration', 'nodule', 'calcification', 'normal', 'pneumonia'],
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))



# In[5]:



# model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[6]:


num_training_images = 48800
num_valid_images = 16266
batch_SIZE = 2
epochs = 30
print("Starting to train>>>>")
Y_pred = model.evaluate_generator(generator=valid_generator,
    steps=num_valid_images//batch_SIZE)
# Y_pred_v2 = model.predict_generator(valid_generator, verbose = True)


# In[ ]:


import math
nb_test_samples = 16266
steps = math.ceil(nb_test_samples/batch_SIZE)

def get_data_with_predictions(model,generator,steps,classes):
    X_test = np.zeros((0,img_height,img_width,3))
    y_test = np.zeros((0,classes))
    preds = np.zeros((0,classes))
    step_count = 0
    for batch_x, batch_y in generator:
        if step_count < steps:
            batch_preds = model.predict(batch_x)
            preds = np.vstack((preds,batch_preds))
            X_test = np.vstack((X_test,batch_x))
            y_test = np.vstack((y_test,batch_y))
            step_count = step_count + 1
        else:
            break
    return X_test, y_test, preds

num_classes=21
X_test, y_test, preds = get_data_with_predictions(model,valid_generator,steps,num_classes)
print(X_test.shape)
print(y_test.shape)
print(preds.shape)

save_array('/home/Erdal.Genc/covid_work/padchest_weights/padchest-X_test.bc', X_test)
save_array('/home/Erdal.Genc/covid_work/padchest_weights/padchest-y_test.bc', y_test)
save_array('/home/Erdal.Genc/covid_work/padchest_weights/padchest-multiclass-preds.bc', preds)

# In[ ]:



# In[1]:


# from vis.utils import utils
# from vis.visualization import visualize_saliency
# from tensorflow.keras import activations
# import matplotlib.image as mpimg
# import scipy.ndimage as ndimage


# img_one = '/mnt/dsets/ChestXrays/PadChest/image_zips/0/99744230716892055301280916536204938895_oo9nk5.png'
# img = mpimg.imread(img_one)
# layer_idx = utils.find_layer_idx(model, 'fc21')
# model.layers[layer_idx].activation = activations.linear
# # model = utils.apply_modifications(model, custom_objects=None)
# grads =  visualize_saliency(model, layer_idx, filter_indices=class_idx, 
#                             seed_input=img, backprop_modifier='guided') 


# In[119]:


# from vis.utils import utils
# from vis.visualization import visualize_saliency
# from tensorflow.keras import activations
# import matplotlib.image as mpimg
# import scipy.ndimage as ndimage


# img_one = '/mnt/dsets/ChestXrays/PadChest/image_zips/0/99744230716892055301280916536204938895_oo9nk5.png'
# img = mpimg.imread(img_one)
# layer_idx = utils.find_layer_idx(model, 'fc21')
# model.layers[layer_idx].activation = activations.linear
# model = utils.apply_modifications(model, custom_objects=None)
# grads =  visualize_saliency(model, layer_idx, filter_indices=class_idx, 
#                             seed_input=img, backprop_modifier='guided') 

