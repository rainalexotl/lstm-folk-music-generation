import os
import re

directory = './datasets/nottingham_database/'
target_path = './datasets/nottingham_database/nottingham_parsed.txt'

def parse(directory):
	"""
	removes chords from abc data as well as all metadata (excluding meter (M) and key (K))
	"""
	abc_data = ""

	# only the meter and key are kept
	meta = re.compile('[MK]:.*?')
	misc = re.compile('%.*?|([A-JLN-Z]:.*?)')

	for file in os.listdir(directory):
		with open(directory + file, 'r') as reader:
			line = reader.readline()
			while (line != ''):
				if (meta.match(line) != None):
					abc_data = abc_data + line
				elif (len(line) > 0 and misc.match(line) == None):
					abc_data = abc_data + re.sub('".*?"', '', line)
				
				# add line but remove chord
				line = reader.readline()

			reader.close()

	f = open(target_path, 'w+')
	f.writelines(abc_data)
	f.close()

parse(directory)