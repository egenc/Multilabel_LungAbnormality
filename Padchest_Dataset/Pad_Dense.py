#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob, time, os, cv2, csv

try:
    get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
    get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=7')
except NameError:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"


from tqdm import tqdm
import shutil
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, optimizers, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import ZeroPadding2D, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Activation, Flatten, Dropout, SpatialDropout2D, MaxPooling2D, GlobalMaxPooling2D, Input
from tensorflow.keras.models import Sequential, Model, model_from_json, model_from_yaml
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

BATCH_SIZE = 8
df_path = '/home/Erdal.Genc/covid_work/PadChest_csv/Padchest_filtered_withPath.csv'
img_path = '/mnt/dsets/ChestXrays/PadChest/image_zips'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())  ## to use which GPU in use
data_copy=pd.read_csv(df_path, low_memory=False)

img_height=224 #299
img_width=224 #299


# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=data_copy,
directory="/mnt/dsets/ChestXrays/PadChest/image_zips",
x_col="Path",
y_col='new_labels',
subset="training",
batch_size=BATCH_SIZE,
seed=42,
shuffle=True,
classes = ['fibrosis', 'pleural_mass', 'pleural_thickening', 'mass', 'fissure', 
           'pneumothorax', 'costophrenic_angle_blunting', 'atelectasis', 'granuloma', 
           'other_lung_abnormality', 'interstitial_pattern', 'emphysema', 'consolidation', 
           'pleural_effusion', 'pulmonary_edema', 'ground_glass_pattern', 
           'infiltration', 'nodule', 'calcification', 'normal', 'pneumonia'],
class_mode="categorical",
target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
dataframe=data_copy,
directory="/mnt/dsets/ChestXrays/PadChest/image_zips",
x_col="Path",
y_col='new_labels',
subset="validation",
batch_size=BATCH_SIZE,
classes = ['fibrosis', 'pleural_mass', 'pleural_thickening', 'mass', 'fissure', 
           'pneumothorax', 'costophrenic_angle_blunting', 'atelectasis', 'granuloma', 
           'other_lung_abnormality', 'interstitial_pattern', 'emphysema', 'consolidation', 
           'pleural_effusion', 'pulmonary_edema', 'ground_glass_pattern', 
           'infiltration', 'nodule', 'calcification', 'normal', 'pneumonia'],
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))


# BATCH_SIZE = 32
# IMG_DIM = 224
# NB_CLASSES = 21
# ds_train = tf.data.Dataset.from_generator(lambda: train_generator,
#                      output_types=(tf.float32, tf.float32),
#                      output_shapes=([BATCH_SIZE, IMG_DIM, IMG_DIM, 3],
#                                     [BATCH_SIZE, NB_CLASSES])).repeat()
#
# ds_valid = tf.data.Dataset.from_generator(lambda: valid_generator,
#                      output_types=(tf.float32, tf.float32),
#                      output_shapes=([BATCH_SIZE, IMG_DIM, IMG_DIM, 3],
#                                     [BATCH_SIZE, NB_CLASSES])).repeat()

# In[4]:


def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
# In[5]:


def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# In[7]:

def ResNet50(input_shape=(224, 224, 3), classes=21):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

# In[8]:
model= ResNet50(input_shape=(img_height, img_width,3), classes= 21)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# In[9]:






num_training_images = 48800
num_valid_images = 16266
epochs = 30
print("Starting to train>>>>")
model.fit_generator(generator=train_generator,
                    steps_per_epoch = num_training_images//BATCH_SIZE,
                    validation_data=valid_generator,
                    validation_steps=num_valid_images//BATCH_SIZE,
                    epochs=epochs
)
model_path = '/home/Erdal.Genc/covid_work/Padchest_work/saved'
model.save(model_path)
# In[ ]:




