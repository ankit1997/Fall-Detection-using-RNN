import os
import pickle
import random
import numpy as np
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt

class Data:
	def __init__(self, path):

		assert '_' not in path, "Please rename {} to replace underscores '_'".format(path)

		self.X = []
		self.Y = []
		self.seq_len = []
		subjects = self._getdirs(path)

		fallDirs = []
		nonFallDirs = []
		for subject in subjects:
			dirs = self._getdirs(subject)
			for d in dirs:
				if "ADL" in d:
					nonFallDirs.append(d)
				elif "FALLS" in d:
					fallDirs.append(d)
				else:
					print("Unreachable code.")

		fallFiles = self._get_class_files(fallDirs)
		nonFallFiles = self._get_class_files(nonFallDirs)
		files = fallFiles + nonFallFiles
		files.sort()

		prefixes = list(set([x.split('_')[0] for x in files]))
		batches = []
		for pref in prefixes:
			batches.append([x for x in files if x.startswith(pref)])

		for i in tqdm(range(len(batches))):
			batch = batches[i]
			n = len(batch)
			trails = n//3
			class_ = 0 if "ADL" in batch[0] else 1

			for i in range(trails):
				acc = batch[i]
				gyro = batch[i+trails]
				ori = batch[i+2*trails]

				data = self._read_trail(acc, gyro, ori)
				self.X.append(data)
				self.Y.append([class_]*len(data))
				self.seq_len.append(data.shape[1])
		
		self.X = np.concatenate(self.X)
		self.Y = np.concatenate(self.Y)
		self.Y = self._to_onehot(self.Y)
		self.seq_len = np.array(self.seq_len)

		# Shuffle
		indices = list(range(self.X.shape[0]))
		random.shuffle(indices)

		self.X = self.X[indices]
		self.Y = self.Y[indices]
		self.seq_len = self.seq_len[indices]

	def _read_trail(self, acc, gyro, ori):
		'''
			Reads a trail given acc, gyro and ori sensor info files.
		'''
		acc_data = self._read_file(acc)
		gyro_data = self._read_file(gyro)
		ori_data = self._read_file(ori)

		# M = max(len(acc_data), len(gyro_data), len(ori_data))
		M = 6000
		acc_data = self._adjust_len(acc_data, M)
		gyro_data = self._adjust_len(gyro_data, M)
		ori_data = self._adjust_len(ori_data, M)

		combine = []
		for i in range(len(acc_data)):
			combine.append(acc_data[i] + gyro_data[i] + ori_data[i])
		combine = np.array(combine)

		return np.expand_dims(combine, 0)

	def _read_file(self, fname):
		data = open(fname).read().strip().split('\n')
		try:
			start = data.index('@DATA')+1
		except ValueError:
			print("Improper header information in {}".format(fname))
			exit()

		features = []
		data = data[start: ]
		for line in data:
			cols = line.split(',')
			cols = list(map(float, cols))
			features.append(cols[1:]) # remove timestamp information

		# print(np.asarray(features).shape)

		return features

	def _adjust_len(self, data, max_len):
		if len(data) <= max_len:
			return data + [[0.0, 0.0, 0.0]]*(max_len-len(data))
		else:
			return data[:max_len]

	def _get_class_files(self, directories):
		files = []
		for directory in directories:
			dirs = self._getdirs(directory)
			for d in dirs:
				files.extend(self._get_files(d))
		assert all([f.endswith('.txt') for f in files]), "Unexpected file extension found"
		return files

	def _getdirs(self, path):
		return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

	def _get_files(self, path):
		return [os.path.join(path, d) for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]

	def _to_onehot(self, array):
		rows = array.shape[0]
		cols = len(np.unique(array))
		temp = np.zeros((rows, cols))
		temp[range(rows), array] = 1
		return temp

class DataLoader:
	def __init__(self, path, batch_size=16, training=True, train_size=0.8, valid_size=0.1, test_size=0.1):
		assert (train_size+valid_size+test_size) == 1.0, "Incorrect train/valid/test split."

		self.data = self._get_data(path)
		self.batch_size = batch_size
		n_points = self.data.X.shape[0]

		train_ind = int(n_points*train_size)
		valid_ind = int(n_points*(train_size+valid_size))
		test_ind = 1.0

		self.train_X = self.data.X[:train_ind]
		self.train_Y = self.data.Y[:train_ind]
		self.train_spe = ceil(self.train_X.shape[0] / self.batch_size) # steps per epoch

		self.valid_X = self.data.X[train_ind: valid_ind]
		self.valid_Y = self.data.Y[train_ind: valid_ind]
		self.valid_spe = ceil(self.valid_X.shape[0] / self.batch_size) # steps per epoch

		self.test_X = self.data.X[valid_ind:]
		self.test_Y = self.data.Y[valid_ind:]
		self.test_spe = ceil(self.test_X.shape[0] / self.batch_size) # steps per epoch

	def train_generator(self):
		i = 0
		while True:
			start = i*self.batch_size
			end = min(start+self.batch_size, self.train_X.shape[0])
			yield self.train_X[start: end], self.train_Y[start: end]

	def valid_generator(self):
		i = 0
		while True:
			start = i*self.batch_size
			end = min(start+self.batch_size, self.valid_X.shape[0])
			yield self.valid_X[start: end], self.valid_Y[start: end]

	def test_generator(self):
		i = 0
		while True:
			start = i*self.batch_size
			end = min(start+self.batch_size, self.test_X.shape[0])
			yield self.test_X[start: end], self.test_Y[start: end]

	def _get_data(self, path):
		fname = path+'.pkl'
		if os.path.isfile(fname):
			print("NOTE: Using preloaded data...")
			return pickle.load(open(fname, 'rb'))
		
		data = Data(path)
		pickle.dump(data, open(fname, 'wb'))
		return data

if __name__ == "__main__":
	data = Data("MobiFallDatasetv2.0")
	print(data.X.shape, data.Y.shape)