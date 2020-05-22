#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wave
import sys, os
import struct
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def get_amps(fname, chunk=1024, plot=False):
    res = list()
    
    wf = wave.open(fname, 'rb')
    data = wf.readframes(chunk)
    
    while len(data) == chunk*2:
        #print(len(data),chunk)
        data_int16 = struct.unpack(str(chunk) + 'h', data)
        data_ar = np.array(data_int16)
        #res.append(np.mean(data_ar))
        res.append(np.max(data_ar) - np.min(data_ar))
        
        data = wf.readframes(chunk)
        
    if plot:
        plt.plot(res)
        
    return np.array(res)

def get_ROI(fname, amp_thres=10000, window=100, win_thres=0.05, chunk=1024, plot=False):
    amps = list()
    
    # read in data
    wf = wave.open(fname, 'rb')
    nframes = wf.getnframes()
    fps = wf.getframerate()
    duration = nframes/fps
    
    data = wf.readframes(chunk)
    
    # extract "amplitude"
    while len(data) == chunk*2:
        #print(len(data),chunk)
        data_int16 = struct.unpack(str(chunk) + 'h', data)
        data_ar = np.array(data_int16)
        #res.append(np.mean(data_ar))
        amps.append(np.max(data_ar) - np.min(data_ar))
        
        data = wf.readframes(chunk)
        
    if plot:
        plt.plot(amps)
        
    # go through amps and find the rough region with large amplitude
    # carry out the check every *window* chunks
    amps = np.array(amps)
    indices = np.where(amps > amp_thres)[0]
    
    ROI_bounds = list()
    
    for chunk_ind in range(0, len(amps), window):
        in_window = np.where((indices >= chunk_ind) & (indices < min(chunk_ind + window, len(amps))))[0]
        #print(in_window)
        
        if len(in_window)/window > win_thres:
        #if len(in_window) > 0:
            if len(ROI_bounds) == 0:
                ROI_bounds.append(chunk_ind)
            #elif ROI_bounds[-1] < chunk_ind - window:
            #    ROI_bounds.append(chunk_ind)
            elif (len(ROI_bounds) % 2 == 0) & (ROI_bounds[-1] < chunk_ind - window):
                ROI_bounds.append(chunk_ind)
            else:
                pass  
        else:
            if len(ROI_bounds) % 2 == 1:
                ROI_bounds.append(chunk_ind)
        
    if len(ROI_bounds) % 2 == 1:
        ROI_bounds.append(len(amps))
        
    # also calculate the total secs of "sound" using ROI_secs
    if len(ROI_bounds) > 0:
        ROI_secs = np.array(ROI_bounds) * chunk / fps
        sound_duration = 0
        for i in range(0, len(ROI_secs), 2):
            sound_duration += ROI_secs[i+1] - ROI_secs[i]
    else:
        ROI_secs = []
        sound_duration = 0
        
    
    return ROI_bounds, ROI_secs, sound_duration

if __name__ == '__main__':
    # pass system arguments and get input & output dirs
    args = sys.argv

    if len(args) == 3:
        in_dir = os.path.abspath(args[1])
        out_dir = os.path.abspath(args[2])
    elif len(args) == 2:
        in_dir = os.path.abspath(os.getcwd())
        out_dir = os.path.abspath(args[1])
    else:
        raise Exception("Need 1 or 2 arguments!")

    if not os.path.exists(in_dir):
        raise Exception("The input directory doesn't exist!")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #print(in_dir, out_dir)

    # get files in in_dir:
    files = os.listdir(in_dir)
    files = [f for f in files if f.endswith('.wav')]
    files.sort()

    print("Processing {} files in total.".format(len(files)))

    # get baseline stuff first
    f_base = os.path.join(in_dir, files[0])
    base_amps = get_amps(f_base)

    # get the 90% quantile of amps from the baseline
    AMP_THRES = np.quantile(base_amps, 0.9)
    WINDOW = 200
    WIN_THRES = 0.05

    # process rest of the files
    total_sound_duration = 0
    total_ROIs = dict()

    for f in files[1:]:
        fpath = os.path.join(in_dir, f)
        __, ROI_secs, sound_duration = get_ROI(fpath, amp_thres=AMP_THRES, 
            window = WINDOW, win_thres = WIN_THRES)

        if sound_duration > 0:
            total_sound_duration += sound_duration
            total_ROIs[f] = ROI_secs

    # output results
    room_date = in_dir.split('/')[-1]
    print('For {}, estimated duration of sound is {:.2f} seconds'.format(room_date,total_sound_duration))

    # save results
    if total_sound_duration > 0:
        total_ROIs['total'] = total_sound_duration
        out_path = os.path.join(out_dir, room_date+'.pkl')
        pkl.dump(total_ROIs, file=open(out_path,'wb'))
        print('Results saved to'+out_path)












