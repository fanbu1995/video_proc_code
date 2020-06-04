#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:00:55 2020

@author: fan
"""

#%%
# Script to go through video files in a directory and do 
# 1) store one frame per 60s in each video to the given image directory
# 2) wrangle MASK RCNN detection results to a "det.txt" file
# 3) run deep_sort generate_detection.py to generate detection and save it somewhere
# 4) run deep_sort deep_sort_app.py to run the tracker and save the output
# 5) post process the tracker output file to calculate daily movement and store the result somewhere

#%%
from deep_sort_utils import *
from rename_dates import date2num
import subprocess

#%%
# set data root dir
data_root = os.path.abspath('../data/')

# set results file root dir
save_root = os.path.abspath('../filetransfer/results1/')

# set directory to store temporary results to use for deep sort detections
temp_dir = os.path.abspath('./temp')
if not os.path.exists(temp_dir):
	os.makedirs(temp_dir)

img_dir = os.path.join(temp_dir,"temp/img1/")
if not os.path.exists(img_dir):
	os.makedirs(img_dir)
det_dir = os.path.join(temp_dir,"temp/det/")
if not os.path.exists(det_dir):
	os.makedirs(det_dir)

# set directory for deep sort detection file
detection_dir = os.path.abspath('./detections')
if not os.path.exists(detection_dir):
	os.makedirs(detection_dir)

# set directory for deep sort output
output_dir = os.path.abspath('./ds_output')
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# a dictionary to save results on estimated daily movements
movements = {'room':[], 'date': [], 'total_movement':[], 
             'minutes_elapsed': [], 'indiv_movement':[]}

# get all week folders
weeks_root = os.listdir(data_root)
weeks_root = [os.path.join(data_root,wr) for wr in weeks_root if wr.startswith('Week')]

rooms_root = ['room230/', 'room324/', 'room351/','room352/']


for wr in weeks_root:

	for rm in rooms_root:
		room_number = rm[4:7]
        
		week_room_dir = os.path.abspath(os.path.join(wr, rm))
		#print(week_room_dir)
		#print(os.listdir(week_room_dir))
		
		# skip room 230, Week 7
		#if ('230' in rm) & ('Week7' in wr):
		#	continue 

		save_rm_dir = os.path.join(save_root,rm)
		if not os.path.exists(save_rm_dir):
			os.makedirs(save_rm_dir)
			print('This path is created:',save_rm_dir)

		dates = os.listdir(week_room_dir)
		for d in dates:

			room_date_dir = os.path.abspath(os.path.join(week_room_dir,d))
			#print('Processing directory {}'.format(room_date_dir))

			# remove the space in date name
			if ' ' in d:
				d = ''.join(d.split())

			save_rd_dir = os.path.join(save_rm_dir, date2num(d.lower()))
			if not os.path.exists(save_rd_dir):
				#os.makedirs(save_rd_dir)
				#print('This path is created:', save_rd_dir)
				OSError("The result path {} doesn't exist!!".format(save_rd_dir))
                
			if 'LRV' in os.listdir(room_date_dir):
				video_dir = os.path.join(room_date_dir,"LRV/")
			elif '100GOPRO' in os.listdir(room_date_dir):
				video_dir = os.path.join(room_date_dir,"100GOPRO/","LRV/")
			else:
				continue
							
			#print("Processing video files in:", video_dir)
            
			retcode = frame_dir_video(video_dir, img_dir, save_rd_dir, every_n_seconds = 60, verbose = True)
            
			if retcode==0:
                		# if somehow the directory failed, skip this one entirely and move on
                		print("Processing files in {} failed! Moving on to the next directory.".format(video_dir))
                		continue
            
            		# if saving frames is done, wrangle the detections
			wrangle_res = wrangle_batch_detections(save_rd_dir, output_path=det_dir)
            
			# if there are images under img_dir and there exists det.txt under det_dir
            		# run generate_detection.py
			images=os.listdir(img_dir)
			if len(images) > 0 and os.path.exists(os.path.join(det_dir,'det.txt')):
                
				command_detection = "python3 deep_sort/tools/generate_detections.py --model=resources/networks/mars-small128.pb --mot_dir={} --output_dir={}".format(temp_dir, detection_dir)
                
				command_detection_list = command_detection.split(" ")
                
				completeProc = subprocess.run(command_detection_list)
                
				if completeProc.returncode == 0:
                    			# if succeed, continue running the tracker
					command_tracker = "python3 deep_sort/deep_sort_app.py --sequence_dir={}/temp --detection_file={}/temp.npy --min_confidence=0.3 --nn_budget=100 --display=False --output_file={}/{}.txt".format(temp_dir, detection_dir, output_dir, room_number+'_'+d.lower())
                    
					command_tracker_list = command_tracker.split(" ")
					completeProc2 = subprocess.run(command_tracker_list)
					if completeProc2.returncode == 0:
                        			# report the happy news
						print("Successfully run detections and tracker on videos in directory {}!".format(video_dir))
                        
                        			# then post process the output tracking file
						output_path = os.path.join(output_dir, room_number+'_'+d.lower()+'.txt')
						try:
							total_movement, minutes_elapsed, indiv_movement = process_deepsort_output(output_path)
   							# print info
							print("On {} in room {}, {} minutes of human activity is captured, with {} pixels of total_movement.".format(d.lower(),room_number, minutes_elapsed, total_movement))
						except:
							# in case something went wrong
							print("Human movement information somehow unavailable for {} in room{}!".format(d.lower(), room_number))
							total_movement = minutes_elapsed = indiv_movement = np.nan	
				
         					# record the results
						movements['room'].append(room_number)
						movements['date'].append(date2num(d.lower()))
						movements['total_movement'].append(total_movement)
						movements['minutes_elapsed'].append(minutes_elapsed)
						movements['indiv_movement'].append(indiv_movement)

						# also: save results to file every time
						pickle.dump(movements, open('daily_movements.pkl','wb'))
						movements_dat = pd.DataFrame(movements)
						movements_dat.to_csv('daily_movements.csv', index=False, index_label=False)
                        
                    
                    				# remove temp.npy
					if os.path.exists(os.path.join(detection_dir,'temp.npy')):
						os.remove(os.path.join(detection_dir,'temp.npy'))
				else:
					print("Generating detections failed on videos in directory {}!".format(video_dir))
                
                		# remove images and det.txt
				os.remove(os.path.join(det_dir,'det.txt'))
				for img in images:
					os.remove(os.path.join(img_dir,img))
                
			else:
				print("Video frames or det.txt file don't exist for directory {}!".format(video_dir))
                

# finally, save the movements data
pickle.dump(movements, open('daily_movements.pkl','wb'))

movements_dat = pd.DataFrame(movements)
movements_dat.to_csv('daily_movements.csv', index = False, index_label=False)
                        
                
                
