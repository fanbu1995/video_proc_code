#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:41:36 2020

@author: fan
"""

#%%
import numpy as np
import pandas as pd
import os
import pickle

import cv2
#import subprocess


#%%
# function to save frames of a video
def frame_one_video(video_dir, video_name, save_dir=None, frame_counter=1, every_n_seconds = 60, verbose=False):
    
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
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, 
                                         "{:06d}.jpg".format(frame_counter)), frame)
                frame_counter += 1
        
    return frame_counter

#%%
# function to save frames of an entire directory
def frame_dir_video(video_dir, save_dir, meta_dir, every_n_seconds = 60, verbose = True):
    
    # 1. open the MASK RCNN results directory (which contains the "meta" file)
    if os.path.exists(os.path.join(meta_dir,'meta_data.pkl')):
        meta = pickle.load(open(os.path.join(meta_dir,'meta_data.pkl'),'rb'))
    else:
        # if meta_data file doesn't even exist, don't process this directory at all
        return 0
    
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
                
                frame_counter = frame_one_video(video_dir, video_name, save_dir, frame_counter, 60, verbose=False)
                
    
    if verbose:
        print("Done processing and saved {} total frames! Frames saved to {}".format(frame_counter-1, save_dir))
        
    return 1

#%%

# a function to convert a single Mask RCNN detection result list

def wrangle_detections(results, output_path=None):
    '''
    wrangle RCNN detections into the format of MOT Challenge
    results: LIST of frame-wise detection results; should be res['info'] of each pickle file
    output_path: path to the output "det.txt" file
    '''
    if (results is None) or (len(results) == 0):
        return
    
    out = {'frame_id':[], 'confidence':[],
           'b_left':[], 'b_top':[], 'b_width':[], 'b_height':[]}
    
    for frame, r in enumerate(results):
        # if entry empty, skip
        if (len(r)==0) or (len(r['class_ids'])==0):
            continue
        # if no people detected, skip
        if not np.any(r['class_ids']==1):
            continue
        # otherwise, proceed
        IDs = r['class_ids']
        ROIs = r['rois'][IDs==1]
        Scores = r['scores'][IDs==1]
        
        out['b_left'].extend(list(ROIs[:,1]))
        #out['b_top'].extend(list(ROIs[:,2]))
        out['b_top'].extend(list(ROIs[:,0]))
        out['b_width'].extend(list(ROIs[:,3]-ROIs[:,1]))
        out['b_height'].extend(list(ROIs[:,2]-ROIs[:,0]))
        out['confidence'].extend(list(Scores))
        
        n_people = len(Scores)
        out['frame_id'].extend([frame+1]*n_people)
        
    # make it into a data frame
    out_dat = pd.DataFrame(out)
    N = out_dat.shape[0]
    out_dat['track_id'] = [-1] * N
    
    out_dat = out_dat[['frame_id','track_id','b_left','b_top','b_width','b_height','confidence']]
    out_dat['x'] = [-1] * N
    out_dat['y'] = [-1] * N
    out_dat['z'] = [-1] * N
    
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out_dat.to_csv(os.path.join(output_path,"det.txt"), header=False, 
                   index = False, index_label=False)
        
        return
    else:
        return out_dat

#%%
# test this
#res = pickle.load(open('/Users/fan/Downloads/lab-meeting-results.pkl','rb'))
#wrangle_detections(res['info'])

# it works

#%%

# a function to batch process the conversion
def wrangle_batch_detections(input_path, output_path=None):
    '''
    Batch process detection results in a directory and 
    then output a single det.txt under output_path
    Returns the dataframe of detection results if output_path=None
    (the detection results might be None, though.)
    '''
    files = os.listdir(input_path)
    if len(files)==0:
        print('Input directory is empty!')
        return
    
    
    # read in meta_data
    if 'meta_data.pkl' in files:
        meta = pickle.load(open(os.path.join(input_path,'meta_data.pkl'),'rb'))
    else:
        meta = None
        
    frame_counter = 0
    all_dat = None
    if meta is not None:
        # loop over meta data to decide whether to skip a video segment
        for item in meta:
            fpath = os.path.join(input_path,item['video_name'].split(".")[0]+'.pkl')
            skip = item['skip']
            try:
                # just in case that video wasn't processed
                res = pickle.load(open(fpath,'rb'))
                if skip == 0:
                    # if not skip, extract detection results
                    # add on the accumulated frame counts
                    # and append to existing data set
                    dat = wrangle_detections(res['info'],output_path=None)
                    dat['frame_id'] += frame_counter
                    all_dat = pd.concat([all_dat,dat])
            except:
                res = None
            # add frame counts if there is a pickle file for that video
            if res:
                frame_counter += len(res['times'])
    else:
        files = [f for f in files if f.endswith('pkl')]
        files.sort()
        # loop over the detection results in order to extract info
        for f in files:
            fpath = os.path.join(input_path,f)
            res = pickle.load(open(fpath,'rb'))
            dat = wrangle_detections(res['info'],output_path=None)
            dat['frame_id'] += frame_counter
            all_dat = pd.concat([all_dat,dat])
            frame_counter += len(res['times'])
        
    # create output_path and save results
    if (output_path is not None) and (all_dat is not None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        all_dat.to_csv(os.path.join(output_path,"det.txt"), header=False, 
                       index = False, index_label=False)
    
        return
    else:
        return all_dat
    
#%%
        
# try this out    
#wrangle_batch_detections('/Users/fan/Downloads/try_wrangle') 
# it works  
        
    
#%%
# function to generate detections and run the tracker on a directory's video
#def detect_and_track(mot_dir, detection_dir, output_dir, out_fname='tracking.txt'):
#    '''
#    mot_dir: the directory that has "img1" and "det" folders as deep sort input
#    detection_dir: the directory for storing detection file (the .npy file)
#    output_dir: the path to output tracking results
#    out_fname: output file name; default 'tracking.txt'
#    '''
#    return
#    
#%%
# post process deep sort output 
def process_deepsort_output(fpath, minutes_per_frame=1):
    '''
    Takes in the path to the output file
    and returns 
        - the total amount of movement for all people
        - total time elasped (in minutes) across all frames (for calculating average later on)
        - the total amount of movement for each person (as a dictionary of id: movement)
    Note: the "output file" should be a CSV style file of 10 columns (in MOT16 convention)
    ''' 
    dat = np.loadtxt(fpath, delimiter=",")
    N = dat.shape[0]
    
    minutes_elapsed = (dat[:,0].max() - dat[:,0].min()) * minutes_per_frame
    
    IDs = np.unique(dat[:,1])
    centers = np.empty((N,2))
    # assume these caculations are correct... (Not exactly sure where the X and Y axis start in an image)
    centers[:,0] = dat[:,2] + 1/2 * dat[:,4] 
    centers[:,1] = dat[:,3] + 1/2 * dat[:,5]
    
    total_movement = 0
    indiv_movement = dict()
    
    for ID in IDs:
       centers_ID = centers[dat[:,1]==ID,:]
       if centers_ID.shape[0] <= 1:
           # only detected once; assumed no movement
           indiv_movement[int(ID)] = 0
       else:
           # detected multiple times: calculate cumulative distance
           coord_diff = np.diff(centers_ID, axis=0)
           movement = np.apply_along_axis(np.linalg.norm, 1, coord_diff).sum()
           indiv_movement[int(ID)] = movement
           total_movement += movement
    
    return total_movement, minutes_elapsed, indiv_movement
    
    
 #%%
# try it out
# total, minutes, indiv = process_deepsort_output('/Users/fan/Downloads/Lab-Meeting-2-tracking.txt')
    
