#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:37:34 2020

@author: fan
"""

# write code to extract video frames with deep_sort ID and detected bounding box
# for human labeling purposes

#%%
import numpy as np
import pandas as pd
import os
import pickle

import cv2


#%%

# function to save frames of a video
def extract_one_video(video_dir, video_name, frame_bbox, save_dir=None, 
                      frame_counter=1, every_n_seconds = 60, verbose=False):
    
    '''
    frame_bbox: a dictionary of "frame number": bounding box vector (x, y, w, h)
    
    '''
    
    try:
        vs = cv2.VideoCapture(os.path.join(video_dir, video_name))
    except:
        print("Video at {} not accessible! Skipping this one...".format(os.path.join(video_dir, video_name)))
        return
    
    NUM_FRAMES = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    FPS = vs.get(cv2.CAP_PROP_FPS)
    if float(FPS) != 0:
        TOTAL_SEC = float(NUM_FRAMES) / float(FPS)
        TOTAL_MIN = TOTAL_SEC / 60
    else:
        print("Video {} has FPS = 0. Skipping this one...".format(video_name))
        return

    # some general info
    if verbose:
        print("Video has {} total frames, with {} fps; duration is {} seconds, i.e., {} minutes.".format(NUM_FRAMES, FPS, TOTAL_SEC, TOTAL_MIN))

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Directory {} is created!".format(save_dir))

    vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
    CUR_FRAME = vs.get(cv2.CAP_PROP_POS_FRAMES)
    CUR_SEC = 0
    
    # skip the frame at time=0
    grabbed, frame = vs.read()
    
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
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # save this frame to save_dir if given
            # ONLY save the frame if frame in frame_bbox
            if save_dir is not None and frame_counter in frame_bbox:
                
                ## get bounding box parameters and draw bounding box
                x,y,w,h = frame_bbox[frame_counter]
                pt1 = int(x), int(y)
                pt2 = int(x + w), int(y + h)
                cv2.rectangle(frame, pt1, pt2, (0,0,255), 3)
                
                ## save the image to "save_dir"
                cv2.imwrite(os.path.join(save_dir, 
                                         "{:03d}.jpg".format(frame_counter)), frame)
                frame_counter += 1
        
    return frame_counter


#%%

# a function that reads in deep_sort output file, extract the frames in which a new ID is tracked
# and saves that frame with the detected bounding box drawn on it
# this function processes a directory (a day for a room)

def extract_dir_video(video_dir, save_dir, meta_dir, output_fpath, every_n_seconds = 60, verbose = True):
    
    '''
    video_dir: directory of the video files to process and extract frames from
    save_dir: the ROOT directory for saving those frames
    output_fpath: the path to the ds_output file (.txt) from Deep Sort tracker
    '''
    
    # 0. try opening the deep sort output .txt file
    #    and get the frames to extract
    #    as well as the big data table (for the human coder's usage)
    try:
        ds_out = np.loadtxt(output_fpath, delimiter=",")
        if ds_out.size == 0:
            # if it's empty somehow
            # print information and return NOTHING
            print('Output file {} is empty! No need to extract frames.'.format(output_fpath))
            return
        
        # get the frame_bbox dictionary
        IDs, rows = np.unique(ds_out[:,1], return_index=True)
        
        frames = ds_out[rows,0]
        boxes = ds_out[rows,2:6]
        
        nframes = len(frames)
        
        frame_bbox = dict(zip(frames,boxes))
        
        # get the big info table
        room_date = os.path.splitext(os.path.split(output_fpath)[-1])[0]
        room, date = room_date.split("_")
        info = pd.DataFrame({'room': room, 
                             'date': date,
                             'frame_number': frames.astype(int),
                             'deep_sort_ID': IDs,
                             'real_ID': np.nan})
    
        # also create the sub-directory for saving frames
        sub_save_dir = os.path.join(save_dir, room, date)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        
    except:
        raise OSError('Cannot open file at {}'.format(output_fpath))
        
    
    # 1. open the MASK RCNN results directory (which contains the "meta" file)
    if os.path.exists(os.path.join(meta_dir,'meta_data.pkl')):
        meta = pickle.load(open(os.path.join(meta_dir,'meta_data.pkl'),'rb'))
    else:
        # if meta_data file doesn't even exist, don't process this directory at all
        # and return NOTHING
        return
    
    # 2. process the video files that are not skipped
    
    if verbose:
        print("processing directory {}...".format(video_dir))
    
    frame_counter = 1
    
    all_in_dir = os.listdir(video_dir)
    all_files = [fname for fname in all_in_dir if (fname.lower().endswith("lrv") | fname.lower().endswith("mp4"))]
    
    for item in meta:
        if item['skip'] == 0:
            video_name = item['video_name'].split(".")[0]
            cands = [fname for fname in all_files if fname.startswith(video_name)]
            if len(cands) > 0:
                video_name = cands[0]
                
                frame_counter_update = extract_one_video(video_dir, video_name, frame_bbox, sub_save_dir, 
                                                         frame_counter, 60, verbose=False)
                
                if frame_counter_update:
                    frame_counter = frame_counter_update
                
                #frame_counter = frame_one_video(video_dir, video_name, save_dir, frame_counter, 60, verbose=False)
                
    if verbose:
        print("Done processing {} total frames and saved {} of them to {}".format(frame_counter-1, nframes, sub_save_dir))
     
    # return the big info table
    return info