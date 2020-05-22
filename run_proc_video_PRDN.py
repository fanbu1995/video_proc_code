
# import utility funcs
from proc_video_dir import *

# set data root dir
data_root = os.path.abspath('../data/')

# set results file root dir
save_root = os.path.abspath('../filetransfer/results1/')

# get all week folders
weeks_root = os.listdir(data_root)
weeks_root = [os.path.join(data_root,wr) for wr in weeks_root if wr.startswith('Week')]

#print(weeks_root)

rooms_root = ['room230/', 'room324/', 'room351/','room352/']

for wr in weeks_root:
	# run after maintenance on 03/10, 1:30pm
	# skip week 1,4,5,7
	#if ('Week1' in wr)|('Week4' in wr)|('Week5' in wr)|('Week7' in wr):
	#	continue

	# run after 2nd interruption (caused by the July 20 files)
	# only process weeks 8 and 10
	if not ('Week8' in wr)|('Week10' in wr):
		continue

	for rm in rooms_root:
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
			# skip room 324, july 10 and 12
			#if d.lower() in ['july10', 'july12']:
			#	continue

			# skip room 230, July 17-19
			if ('230' in rm) & (d in ['July 17','July 18', 'July 19']):
				continue

			# skip all "July 20"
			if d == 'July 20':
				continue

			room_date_dir = os.path.abspath(os.path.join(week_room_dir,d))
			#print('Processing directory {}'.format(room_date_dir))

			# remove the space in date name
			if ' ' in d:
				d = ''.join(d.split())

			save_rd_dir = os.path.join(save_rm_dir, d.lower())
			if not os.path.exists(save_rd_dir):
				os.makedirs(save_rd_dir)
				print('This path is created:', save_rd_dir)
			if 'LRV' in os.listdir(room_date_dir):
				video_dir = os.path.join(room_date_dir,"LRV/")
			elif '100GOPRO' in os.listdir(room_date_dir):
				video_dir = os.path.join(room_date_dir,"100GOPRO/","LRV/")
			else:
				continue
							
			print("Processing video files in:", video_dir)
			process_dir_video(video_dir, save_rd_dir, cpu_count=1, images_per_cpu=12)

										

