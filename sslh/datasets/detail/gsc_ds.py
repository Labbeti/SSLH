
import numpy as np
import os
import os.path as osp
import random
import soundfile
import torch
import torchaudio

from sslh.datasets.detail.gsc_ds_core import SPEECHCOMMANDS

from tqdm import trange
from torch import Tensor
from torch.nn import Module
from typing import Tuple


URL = "speech_commands_v0.02"
EXCEPT_FOLDER = "_background_noise_"

target_mapper = {
	"bed": 0,
	"bird": 1,
	"cat": 2,
	"dog": 3,
	"down": 4,
	"eight": 5,
	"five": 6,
	"follow": 7,
	"forward": 8,
	"four": 9,
	"go": 10,
	"happy": 11,
	"house": 12,
	"learn": 13,
	"left": 14,
	"marvin": 15,
	"nine": 16,
	"no": 17,
	"off": 18,
	"on": 19,
	"one": 20,
	"right": 21,
	"seven": 22,
	"sheila": 23,
	"six": 24,
	"stop": 25,
	"three": 26,
	"tree": 27,
	"two": 28,
	"up": 29,
	"visual": 30,
	"wow": 31,
	"yes": 32,
	"zero": 33,
	"backward": 34,
	"silence": 35
}
all_classes = target_mapper


# =============================================================================
# UTILITY FUNCTION
# =============================================================================


def _split_s_u(train_dataset, s_ratio: float = 1.0):
	_train_dataset = SpeechCommandsStats.from_dataset(train_dataset)

	nb_class = len(target_mapper)
	dataset_size = len(_train_dataset)

	if s_ratio == 1.0:
		return list(range(dataset_size)), []

	s_idx, u_idx = [], []
	nb_s = int(np.ceil(dataset_size * s_ratio) // nb_class)
	cls_idx = [[] for _ in range(nb_class)]

	# To each file, an index is assigned, then they are split into classes
	for i in trange(dataset_size):
		y, _, _ = _train_dataset[i]
		cls_idx[y].append(i)

	# Recover only the s_ratio % first as supervised, rest is unsupervised
	for i in trange(len(cls_idx)):
		random.shuffle(cls_idx[i])
		s_idx += cls_idx[i][:nb_s]
		u_idx += cls_idx[i][nb_s:]

	return s_idx, u_idx


def cache_feature(func):
	def decorator(*args, **kwargs):
		key = ",".join(map(str, args))

		if key not in decorator.cache:
			decorator.cache[key] = func(*args, **kwargs)

		return decorator.cache[key]

	decorator.cache = dict()
	decorator.func = func
	return decorator


class SpeechCommands(SPEECHCOMMANDS):
	def __init__(self,
				 root: str,
				 subset: str = "train",
				 url: str = URL,
				 download: bool = False,
				 transform: Module = None) -> None:
		super().__init__(root, url, download, transform)

		assert subset in ["train", "validation", "testing"]
		self.subset = subset
		self.root_path = self._walker[0].split("/")[:-2]
		if self.root_path[0] == "":
			self.root_path[0] = "/"

		self._keep_valid_files()

	@cache_feature
	def __getitem__(self, index: int) -> Tuple[Tensor, int]:
		waveform, _, label, _, _ = super().__getitem__(index)
		return waveform, target_mapper[label]

	def save_cache_to_disk(self, name) -> None:
		path = os.path.join(self._path, f"{name}_features.cache")
		torch.save(self.__getitem__.cache, path)

	def load_cache_from_disk(self, name) -> bool:
		path = os.path.join(self._path, f"{name}_features.cache")

		if os.path.isfile(path):
			disk_cache = torch.load(path)
			self.__getitem__.cache.update(disk_cache)
			return True

		return False

	def _keep_valid_files(self):
		# bn = os.path.basename
		bn2 = lambda x: "/".join(x.split("/")[-2:])
		# bn2 = lambda x: osp.join(osp.basename(osp.dirname(x)), osp.basename(x))

		def file_list(filename):
			path = os.path.join(self._path, filename)
			with open(path, "r") as f:
				to_keep = f.read().splitlines()
				return set([path for path in to_keep])

		# Recover file list for validaiton and testing.
		validation_list = file_list("validation_list.txt")
		testing_list = file_list("testing_list.txt")

		# Create it for training
		import time
		start_time = time.time()

		training_list = [
			bn2(path)
			for path in self._walker
			if bn2(path) not in validation_list and bn2(path) not in testing_list
		]

		if self.subset == "train":
			for p in training_list:
				if p in validation_list:
					print("%s is train and validation" % p)
					raise ValueError()

				if p in testing_list:
					print("%s is in both train and testing list" % p)
					raise ValueError()

		# print("run in %f" % (time.time() - start_time))

		# Map the list to the corresponding subsets
		mapper = {
			"train": training_list,
			"validation": validation_list,
			"testing": testing_list,
		}

		self._walker = [
			os.path.join(*self.root_path, path)
			for path in mapper[self.subset]
		]


class SpeechCommandsStats(SpeechCommands):
	@classmethod
	def from_dataset(cls, dataset: SPEECHCOMMANDS):
		root = dataset.root

		newone = cls(root=root)
		newone.__dict__.update(dataset.__dict__)
		return newone

	def _load_item(self, filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
		HASH_DIVIDER = "_nohash_"
		relpath = os.path.relpath(filepath, path)
		label, filename = os.path.split(relpath)
		speaker, _ = os.path.splitext(filename)

		speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
		utterance_number = int(utterance_number)

		# remove Load audio
		# waveform, sample_rate = torchaudio.load(filepath)
		# return waveform, sample_rate, label, speaker_id, utterance_number
		return label, speaker_id, utterance_number

	def __getitem__(self, index: int) -> Tuple[Tensor, int]:
		fileid = self._walker[index]

		label, speaker_id, utterance_number = self._load_item(fileid, self._path)

		return target_mapper[label], speaker_id, utterance_number


class SpeechCommand10(SpeechCommands):
	TRUE_CLASSES = ["yes", "no", "up", "down", "left",
					"right", "off", "on", "go", "stop"]

	def __init__(self,
				 root: str,
				 subset: str = "train",
				 url: str = URL,
				 download: bool = False,
				 transform: Module = None,
				 percent_to_drop: float = 0.5) -> None:
		super().__init__(root, subset, url, download, transform)

		assert 0.0 <= percent_to_drop < 1.0

		self.percent_to_drop = percent_to_drop

		self.target_mapper = {
			"yes": 0,
			"no": 1,
			"up": 2,
			"down": 3,
			"left": 4,
			"right": 5,
			"off": 6,
			"on": 7,
			"go": 8,
			"stop": 9,
			"silence": 10,
			"unknown": 11  # _background_noise_
		}

		# the rest of the classes belong to the "junk / trash / poubelle class"
		for cmd in all_classes:
			if cmd not in self.target_mapper:
				self.target_mapper[cmd] = 11

		self.drop_some_trash()
		self.add_silence()

	@cache_feature
	def __getitem__(self, index: int) -> Tuple[Tensor, int]:
		filepath = self._walker[index]

		label = filepath.split("/")[-2]
		target = self.target_mapper[label]

		waveform, _ = super().__getitem__(index)

		return waveform, target

	def drop_some_trash(self):
		def is_trash(path: str) -> bool:
			return self.target_mapper[path.split("/")[-2]] == 11

		# Create the complete list of trash class
		trash_list = [path for path in self._walker if is_trash(path)]

		# choice only x% of it that will be removed
		nb_to_drop = int(len(trash_list) * self.percent_to_drop)
		to_drop = np.random.choice(trash_list, size=nb_to_drop, replace=False)

		# remove it from the _walker
		self._walker = list(set(self._walker) - set(to_drop))

		print("%d out of %s junk files were drop." % (len(to_drop), len(trash_list)))

	def add_silence(self):
		"""For the class silence, a new directory is created called "silence"
		It will contain 1 seconds segment from the _background_noise_ directory
		If the directory already exist, do some verification and pass
		"""
		silence_dir = os.path.join(*self.root_path, "silence")

		if os.path.isdir(silence_dir):
			self._check_silence_class()

		else:
			self._create_silence_class()
			self._check_silence_class()

	def _create_silence_class2(self):
		print("Silence class doesn't exist")
		silence_dir = os.path.join(*self.root_path, "silence")
		noise_path = os.path.join(*self.root_path, "_background_noise_")

		# the silence class directory doesn't exist, create it
		os.makedirs(silence_dir)

		# Split each noise files into 1 second long segment
		to_process = []
		for file in os.listdir(os.path.join(*self.root_path, EXCEPT_FOLDER)):
			if file[-4:] == ".wav":
				to_process.append(os.path.join(noise_path, file))

		# Basic way, split each files into 1 seconds long segment
		print("Creating silence samples...")
		for filepath in to_process:
			basename = os.path.basename(filepath)

			waveform, sr = torchaudio.load(filepath)

			nb_full_segment = int(len(waveform[0]) / sr)
			rest = len(waveform[0]) % sr
			segments = np.split(waveform[0][:-rest], nb_full_segment)

			# write each segment as a wav file with a unique name
			for i, s in enumerate(segments):
				unique_id = f"{basename[:-4]}_nohash_{i}.wav"
				path = os.path.join(silence_dir, unique_id)
				soundfile.write(path, s, sr)
		print("done")

	def _create_silence_class(self):
		print("Silence class doesn't exist")
		silence_dir = os.path.join(*self.root_path, "silence")
		noise_path = os.path.join(*self.root_path, "_background_noise_")

		# the silence class directory doesn't exist, create it
		os.makedirs(silence_dir)

		# Split each noise files into 1 second long segment
		to_process = []
		for file in os.listdir(os.path.join(*self.root_path, EXCEPT_FOLDER)):
			if file[-4:] == ".wav":
				to_process.append(os.path.join(noise_path, file))

		# Basic way, split each files into 1 seconds long segment
		print("Creating silence samples...")
		for filepath in to_process:
			basename = os.path.basename(filepath)

			waveform, sr = torchaudio.load(filepath)

			# write each segment, we will create 300 segments of 1 secondes
			start_timestamps = np.random.randint(0, len(waveform[0]) - sr, size=400)
			for i, st in enumerate(start_timestamps):
				unique_id = f"{basename[:-4]}_nohash_{i}.wav"
				path = os.path.join(silence_dir, unique_id)

				segment = waveform[0][st:st + sr]
				soundfile.write(path, segment, sr)

		print("done")

	def _check_silence_class(self):
		silence_dir = os.path.join(*self.root_path, "silence")
		all_files = os.listdir(silence_dir)

		print("Silence class already processed")
		print("%s samples present" % len(all_files))
