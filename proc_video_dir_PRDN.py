#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:56:49 2020

@author: fan
"""

#%%
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import cv2

import pickle

#%%


## CHANGE THIS TO THE ACTUAL DIRECTORY OF MASK RCNN!! ##
# Root directory of the Mask RCNN project
ROOT_DIR = os.path.abspath("/home/fb75/mount/shared/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    

#%%
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


#%%
# function to process one video file
def process_one_video(video_dir, video_name, model, every_n_seconds = 60, bright_threshold = 60, show_instance = False, verbose = False):
    
    # 0. Pre-screen the video: read 1/10, 5/10, 9/10 frames, 
    # if brightness < 0.3, skip the entire video and only return the total length
    try:
        vs = cv2.VideoCapture(os.path.join(video_dir, video_name))
    except:
        print("Video at {} not accessible! Skipping this one...".format(os.path.join(video_dir, video_name)))
        return({'video_name': video_name, 'skip': 1, 'total_sec': None}, None)
    
    NUM_FRAMES = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    FPS = vs.get(cv2.CAP_PROP_FPS)
    if float(FPS) != 0:
        TOTAL_SEC = float(NUM_FRAMES) / float(FPS)
        TOTAL_MIN = TOTAL_SEC / 60
    else:
        print("Video {} has FPS = 0. Skipping this one...".format(video_name))
        return({'video_name': video_name, 'skip':1, 'total_sec': None}, None)

    
    # some general info
    if verbose:
        print("Video has {} total frames, with {} fps; duration is {} seconds, i.e., {} minutes.".format(NUM_FRAMES, FPS, TOTAL_SEC, TOTAL_MIN))
    
    dark = []
    
    for ratio in [0.1, 0.5, 0.9]:
        # vs.set(cv2.CAP_PROP_POS_AVI_RATIO, ratio) # somehow "CAP_PROP_POS_AVI_RATIO" doesn't work??
        vs.set(cv2.CAP_PROP_POS_FRAMES, int(ratio*NUM_FRAMES))
        grabbed, frame = vs.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame_hsv)
        dark.append(np.mean(v) < bright_threshold)
    
    if all(dark):
        return({'video_name': video_name, 'skip': 1, 'total_sec': TOTAL_SEC}, None)

    # 1. Get batch size from model configuration

    batch_size = model.config.BATCH_SIZE
    
    # 2. Video processing
    #vs.set(cv2.CAP_PROP_POS_MSEC, 0)
    #CUR_MSEC = 0
    # the second thing just doesn't work!
    vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
    CUR_FRAME = vs.get(cv2.CAP_PROP_POS_FRAMES)
    CUR_SEC = 0
    
    # skip the frame at time=0
    grabbed, frame = vs.read()
    images = []#; times = []#; num_people = []
    
    All_results = {'times': [], 'info': []}
    
    while grabbed:
        #vs.set(cv2.CAP_PROP_POS_MSEC, CUR_MSEC + every_n_seconds * 1000) # this doesn't work either??
        vs.set(cv2.CAP_PROP_POS_FRAMES, CUR_FRAME + int(every_n_seconds * FPS))
        CUR_FRAME = vs.get(cv2.CAP_PROP_POS_FRAMES)
        
        #CUR_MSEC = vs.get(cv2.CAP_PROP_POS_MSEC)
        #CUR_SEC = float(CUR_MSEC) / float(1000)
        
        CUR_SEC = float(CUR_FRAME) / float(FPS)
        

        grabbed, frame = vs.read()
        if grabbed:
            
            if verbose:
                print("Processing frame {}, approx. {} seconds into the video...".format(CUR_FRAME, CUR_SEC))
        
            # fix color
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            # append image to image list
            images.append(frame_rgb)
            All_results['times'].append(CUR_SEC)
            
            # if reached batch size, run detection
            # and empty the image list
            if len(images) == batch_size:
                #results = model.detect(images, verbose=1)
                results = model.detect(images, verbose=0)
                
                # visualize an instance
                if show_instance: 
                    r = results[0]
                    visualize.display_instances(images[0], r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                
                All_results['info'].extend(results)
                images = []
    
    # at the end: deal with leftover images 
    if((len(images) > 0) & (len(images) < batch_size)):
        # padding with all-zero fake frame
        fake_image = np.zeros_like(frame_rgb)
        num_fake = batch_size - len(images)
        images.extend([fake_image]*num_fake)
        # detection, but only save the real ones
        results = model.detect(images, verbose=1)
        
        # visualize an instance
        if show_instance: 
            r = results[0]
            visualize.display_instances(images[0], r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        
        All_results['info'].extend(results[:-num_fake])
        # empty it again
        images = []
        
    return({'video_name': video_name,'skip': 0, 'total_sec': TOTAL_SEC}, All_results)
    
#%%
# function to process all video files under a directory
# and save results 
def process_dir_video(video_dir, save_dir, cpu_count = 1, images_per_cpu = 4, every_n_seconds = 60, bright_threshold = 60, verbose = True):
    
    # 1. configure the model
    class InferenceConfig(coco.CocoConfig):
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = cpu_count
        IMAGES_PER_GPU = images_per_cpu

    config = InferenceConfig()
    
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    if verbose:
        print("Model loaded!")
    
    
    # 2. process files one by one
    # and save results in pickle
    all_in_dir = os.listdir(video_dir)
    all_files = [fname for fname in all_in_dir if (fname.lower().endswith("lrv") | fname.lower().endswith("mp4"))]
    all_files.sort()
    
    all_meta_data = []

    meta_fpath = os.path.join(save_dir, 'meta_data.pkl')
    
    for video_fname in all_files:
        
        if verbose:
            print("Processing video file {}......".format(video_fname))
        
        meta, res = process_one_video(video_dir, video_fname, model, every_n_seconds, bright_threshold)
    
        all_meta_data.append(meta)
        
        if res:       
            res_fname = "results_"+video_fname.split(".")[0]+".pkl"
            res_fpath = os.path.join(save_dir, res_fname)
            pickle.dump(res, file = open(res_fpath, "wb"))
       
        # save metadata every time, just in case...
        pickle.dump(all_meta_data, file = open(meta_fpath, "wb"))
        
    # 3. save all the meta data
    
    #meta_fpath = os.path.join(save_dir,"meta_data.pkl")
    
    pickle.dump(all_meta_data, file = open(meta_fpath, "wb"))
    
    if verbose:
        print("Processing done and files saved.")
    
    return

#%%
if __name__ == '__main__':
    process_dir_video("/Users/fan/Downloads/video-examples/", "/Users/fan/Downloads/", every_n_seconds = 240)
    
#%%
