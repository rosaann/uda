#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:14:52 2018

@author: zl
"""

import utils
import params
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

def img_pre_process(img, resize_w, resize_h):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
#    print("shape  ", shape)
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    ## Return the image sized as a 4D array
    return np.resize(img, (resize_w, resize_h, params.FLAGS.img_c))

from sklearn.model_selection import train_test_split

#加载epoch1到epoch9的所有视频的图片到img_list中，加载epoch1到epoch9的所有转向数据到steer_list，
def loadVideoAndResize(resize_w, resize_h):
    for epoch_id in range(1,11):    
        vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv').format(epoch_id)
        steer_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_steering.csv').format(epoch_id)
        steer_data = utils.fetch_csv_data(steer_path)
    
        steer_list = steer_data['wheel'] 
    
        frame_count = utils.frame_count(vid_path)
        cap = cv2.VideoCapture(vid_path)
        img_list = []
    
        for frame_id in range(frame_count):
            ret, img = cap.read()       
            img_resized = img_pre_process(img, resize_w, resize_h)
            img_list.append(img_resized)
        
            if frame_id == frame_count - 1 and epoch_id == 9:
                print("img pre shape {}".format(img.shape))
                print("img after shape {}".format(img_resized.shape))
                plt.imshow(img)
                plt.show()
                plt.imshow(img_resized)
                plt.show()
    
   
    #把训练集存到本地
        train_add = 'epoch{:0>2}_{}_{}_preprocess_train.p'.format(epoch_id, resize_w,resize_h)
     #   val_add = 'epoch{:0>2}_{}_{}_preprocess_val.p'.format(epoch_id, resize_w,resize_h)
        pickle.dump((img_list, steer_list), open(train_add, 'wb'))
     #   pickle.dump((img_list_val, steer_list_val), open(val_add, 'wb'))



def load_preprocess_training_batch(batch_id, resize_w, resize_h):
    filename = 'epoch{:0>2}_{}_{}_preprocess_train.p'.format(batch_id, resize_w, resize_h)
    img, steer = pickle.load(open(filename, mode='rb'))

    return img, steer
def load_preprocess_validating_batch(batch_id, resize_w, resize_h):
    filename = 'epoch{:0>2}_{}_{}_preprocess_val.p'.format(batch_id, resize_w, resize_h)
    img, steer = pickle.load(open(filename, mode='rb'))

    return img, steer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing.image import ImageDataGenerator


def nor_output(pre_y):
    pre = pre_y.reshape(1, -1)[0]
    for i, x in enumerate(pre):
        pre[i] = round(x, 4)

    for i,x in enumerate(pre):
        if x > 0:
            if (x-math.floor(x))>=0.75:
                pre[i] = int(x) + 1.0
            elif (x-math.floor(x))>=0.25:
                pre[i] = (int(x) + 0.5)
            else:
                pre[i] = int(x)
        else:
            if (x-int(x))<=-0.75:
                pre[i] = int(x) - 1.0
            elif (x-int(x))<=-0.25:
                pre[i] = (int(x) - 0.5)
            else:
                pre[i] = int(x)
    return pre

def simple_model_1(time_len=1):
    ch, row, col = 3, 64, 64  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
  #  model.add(Lambda(lambda x: x,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode="same"))
    
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
#  model.add(Lambda(nor_output_1))
    sgd = optimizers.SGD(lr=0.00003, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

      

    return model

def simple_model_2(time_len=1):
    ch, row, col = 3, 32, 32  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Conv2D(128,kernel_size =(3,3), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(256,kernel_size =(1,1), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(384,kernel_size =(9,9), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512,kernel_size =(1,1), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
      
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(1))

    sgd = optimizers.SGD(lr=0.00003, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
      

    return model

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
def inceptionV3_model(time_len=1):
# create the base pre-trained model 299 * 299
    base_model = InceptionV3(weights='imagenet', include_top=True)

# add a global spatial average pooling layer
    x = base_model.output
 #   x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
  #  x = Dense(1, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1)(x)

# this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#    model.compile(optimizer='sgd', loss='mean_absolute_error')

# train the model on the new data for a few epochs
    

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
 #   for i, layer in enumerate(base_model.layers):
 #       print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
   # from keras.optimizers import SGD
    sgd = optimizers.SGD(lr=0.00003, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
   # model.fit_generator(...)
    return model

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def restNet_model(time_len=1):
    #224 * 224
    base_model = ResNet50(weights='imagenet')

    x = base_model.output
    predictions = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    
    sgd = optimizers.SGD(lr=0.00003, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
  #  for i, layer in enumerate(base_model.layers):
     #   print(i, layer.name)
    return model

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
from decimal import Decimal
def train_model(video_list, model_func, resize_w, resize_h, ifSave = False, test_img=None, test_y=None):
   # model = model_func()
    model = model_func()
 #   model = KerasRegressor(build_fn=model_func, nb_epoch=epoches, batch_size=batch_size, verbose=0)
    ######
  #  for epoch_idx in range(1, (len(video_list) * epoches) + 1) :
  #      
  #      video_idx = epoch_idx % len(video_list)
  #      if video_idx == 0:
  #          video_idx = len(video_list)
  #      print("train epoch_idx {} video_idx{}".format(epoch_idx, video_idx))
  #      x_train, y_train = load_preprocess_training_batch(video_idx, resize_w, resize_h)
  #      x_train = np.array(x_train).transpose(0, 2, 1, 3)
  #      model.fit(x_train, y_train, nb_epoch = 1, batch_size = batch_size)
    ######
  #  X = np.array([])
  #  Y = []
  #  for video_idx in range(1, len(video_list) + 1):
  #      x_train, y_train = load_preprocess_training_batch(video_idx, resize_w, resize_h)
  #      x_train = np.array(x_train).transpose(0,  1, 2, 3)
  #      #其它是0321, interceptv3:0123
  #      if X.shape == np.array([]).shape:
  #          X = x_train
  #          Y = list(y_train)
  #      else:
  #          X = np.vstack((X, x_train))
  #          Y.extend(list(y_train))
  ########   
    
    val_acc = []
    test_acc = []
    for epoch_idx in range(0, epoches):
        
        x_val = np.array([])
        y_val = []
        for video_idx in range(1, len(video_list) + 1):
            x_train, y_train = load_preprocess_training_batch(video_idx, resize_w, resize_h)
            x_train = np.array(x_train).transpose(model_trans)
            print(model_trans)
            #随机打乱数据，取10%的训练数据做验证集
            img_list_train, img_list_val, steer_list_train, steer_list_val = train_test_split(x_train, np.array(y_train), test_size=0.1, random_state=10) 
        #    datagen = ImageDataGenerator(
        #            featurewise_center=True,
        #            featurewise_std_normalization=True                  
        #            )
        #    datagen.fit(img_list_train)
        #    model.fit_generator(datagen.flow(img_list_train, steer_list_train, batch_size=batch_size),
        #            steps_per_epoch=len(steer_list_train), epochs=1)
            
            model.fit(img_list_train, steer_list_train, nb_epoch = 1, batch_size = batch_size)
            if x_val.shape == np.array([]).shape:
                x_val = img_list_val
                y_val = list(steer_list_val)
            else:
                x_val = np.vstack((x_val, img_list_val))
                y_val.extend(list(steer_list_val))
        val_acc.append(test_model(model, x_val, y_val))
        if test_img != None:
            test_acc.append(test_model(model, test_img, test_y))
  #  if ifSave == True:
    utils.save_model(model)
        
    plt.plot(val_acc,color='b', label='验证准确率')
    plt.plot(test_acc, color='g', label='测试准确率')
    plt.show()
    return model

def test_model(model, img_list_val, y_val):
 #   pre = model.predict(img_list_val,  verbose=0)
    pre = model.predict(img_list_val)
    pre = nor_output(pre)
  #  print (pre)
    score = r2_score(y_val, pre)
    print("score: %.2f " % (score))
    

    if type(y_val) != np.array:
        y_val = np.array(y_val)
    y_val = y_val.reshape(len(list(y_val)), 1)    
    y_val = y_val.reshape(1, -1)[0]
    x = np.arange(1, len(y_val) + 1)
    y = y_val  - pre ;

    plt.plot(x, abs(y) , marker = '.', color = 'b', label = 'steer bias',linewidth=0.1)
    plt.show()
    return score
####################start run 

loadVideoAndResize(299, 299)
seed = 10
#0213 为interception
#0312 为其它
model_trans = (0, 2, 1, 3)
batch_size = 64
epoches = 20

res_w = 299
res_h = 299
x_test, y_test = load_preprocess_training_batch(10, res_w, res_h)   
x_test = np.array(x_test).transpose(model_trans)

model_all = train_model(range(1, 10),  inceptionV3_model, res_w, res_h, ifSave = True, test_img = x_test, test_y = y_test)
test_model(model_all, x_test, y_test)