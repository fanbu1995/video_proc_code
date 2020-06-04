import pickle
import os
import sys
import numpy as np
import pandas as pd
import datetime as dt

save_root = os.path.abspath('/home/fb75/mount/filetransfer/summary/')
results_root = os.path.abspath('/home/fb75/mount/filetransfer/results1/')

meta_name = 'meta_data.pkl'
res_prefix = 'results_'
res_suffix = '.pkl'

rooms_root = ['room230', 'room324', 'room351', 'room352']
teams = ['teamD','teamB','teamA','teamC']

day0 = dt.datetime.strptime('20190527','%Y%m%d')
weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# a function to get week and day from a date
def week_day(date):
	date = '2019'+date
	dayn = dt.datetime.strptime(date,'%Y%m%d')
	day_of_wk = dayn.weekday()
	day = weekdays[day_of_wk]
	wk = (dayn - day0).days//7 + 1
	return(wk, day)

# a function to "trim" indices (getting rid of "spurious" nonzeros at start and end)
def trim_indices(inds, trim_num = 3, trim_thres = 10):
	l = len(inds)
	to_trim = []
	# trim the beginning
	for i in range(min(l-1, trim_num)):
		if inds[i+1] - inds[i] >= trim_thres:
			to_trim.extend(range(i+1))
			#to_trim.append(i)
	# trim the end
	for i in range(l-1, max(0,l-trim_num-1), -1):
		if inds[i] - inds[i-1] >= trim_thres:
			to_trim.extend(range(i,l))
			#to_trim.append(i)

	return([inds[i] for i in range(l) if i not in to_trim])

# data storage for a bit summary table
all_duration = {'week':[], 'day':[], 'teamA':[], 'teamB':[], 'teamC':[], 'teamD':[]}


for rr in rooms_root:
	team = teams[rooms_root.index(rr)]

	save_room_dir = os.path.join(save_root, rr)
	if not os.path.exists(save_room_dir):
		os.makedirs(save_room_dir)
	
	rooms_dir = os.path.join(results_root, rr)
	dates_root = os.listdir(rooms_dir)
	dates_root.sort()

	# only process the few july records for now...
	#dates_root = [dr for dr in dates_root if dr.startswith('july')]

	
	for d in dates_root:
		print("Processing data on {} in {}...".format(d, rr))

		week, day = week_day(d)

		# try processing only two of the weeks for now
		#if not week in [4,5]:
		#if week!=2:
		#	continue		

		dates_dir = os.path.join(rooms_dir, d)
		#all_files = os.listdir(dates_dir)
		#fnames = [fn for fn in all_files if fn.startswith(res_prefix)]
		meta = pickle.load(open(os.path.join(dates_dir,meta_name),'rb'))
		
		time_passed = 0
		times = []
		num_people = []		

		for m in meta:
			skip = m['skip']
			duration = m['total_sec']
			if skip == 1:
				if duration != None:
					time_passed += duration
				times.append(time_passed)
				num_people.append(0)
			else:
				vcode = m['video_name'].split('.')[0]
				res_fname = res_prefix + vcode + res_suffix
				res = pickle.load(open(os.path.join(dates_dir,res_fname),'rb'))
				this_times = res['times']
				this_info = res['info']
				for i in range(len(this_times)):
					times.append(this_times[i]+time_passed)
					if 'class_ids' in this_info[i].keys():
						num_people.append(np.sum(this_info[i]['class_ids']==1))
					else:
						num_people.append(0)
				time_passed += duration

		has_people = [i for i in range(len(num_people)) if num_people[i]>0]

		# trim off some boundary false positives...
		has_people = trim_indices(has_people)
		
		if len(has_people)>0:
			people_time = times[max(has_people)] - times[min(has_people)]
		else:
			people_time = 0
		
		if people_time == 0:
			hours = 0
			print('No people all day!\n')
			#print("{} hours".format(0))
		else:
			minutes = float(people_time)/float(60)
			hours = minutes/float(60)
			print('People present for {:.2f} minutes, i.e., {:.2f} hours.\n'.format(minutes,hours))
			hours = round(hours, 2)

		# update the all_duration data
		if team=='teamA':
			all_duration['week'].append(week)
			all_duration['day'].append(day)

		all_duration[team].append(hours)
		
		# create individual day data
		data = pd.DataFrame({'time':times, 'num_people':num_people})
		
		save_fname = rr+'_'+d+'.csv'
		data.to_csv(os.path.join(save_room_dir, save_fname), index=False, index_label=False)
		
			
		
summs = pd.DataFrame.from_dict(all_duration)
summs.to_csv(os.path.join(save_root,'all_duration.csv'), sep='\t', index=False, index_label=False)
print(summs)
