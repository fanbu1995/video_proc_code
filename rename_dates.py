import os, sys


def date2num(date):

	months_t = ['may','june','july','aug']
	months_n = ['05','06','07','08']
	
	if date[-2].isdigit():
		mon = date[:-2]
		da = date[-2:]
	else:
		mon = date[:-1]
		da = '0'+date[-1]
	
	month = months_n[months_t.index(mon)]

	return month+da

if __name__ == '__main__':
	res_root = os.path.abspath('../filetransfer/results1')
	rooms = os.listdir(res_root)
	for rm in rooms:
		os.chdir(os.path.join(res_root,rm))
		#print(sorted(os.listdir()))
		print('Change date folder names for {}...'.format(rm))
		folders = os.listdir()
		for fo in folders:
			newfo = date2num(fo)
			os.rename(fo, newfo)
		print('Done.\n')
		
