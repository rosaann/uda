#!/usr/bin/env python 
import sys
import os
import time
import subprocess as sp
import itertools
## CV
import cv2
## Model
import numpy as np
import tensorflow as tf
## Tools
import utils
## Parameters
import params ## you can modify the content of params.py
import matplotlib.pyplot as plt
## Test epoch
epoch_ids = [10]
## Load model
#model = utils.get_model()

## Preprocess



## Process video
for epoch_id in epoch_ids:
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    print ('vid_path {}'.format(vid_path))
    assert os.path.isfile(vid_path)
    frame_count = utils.frame_count(vid_path)
    print('frame_count {}'.format(frame_count))
    cap = cv2.VideoCapture(vid_path)

    machine_steering = []

    print('performing inference...')
    time_start = time.time()
    for frame_id in range(frame_count):
        ret, img = cap.read()
        plt.imshow(img)
        assert ret
        ## you can modify here based on your model
        img = utils.img_pre_process(img)
        img = img[None,:,:,:]
        deg = float(model.predict(img, batch_size=1))
        machine_steering.append(deg)

    cap.release()

    fps = frame_count / (time.time() - time_start)
    
    print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))
    
    print('performing visualization...')
    utils.visualize(epoch_id, machine_steering, params.out_dir,
                        verbose=True, frame_count_limit=None)
    
    
