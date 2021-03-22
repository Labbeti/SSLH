"""
	Google Speech Commands (GSC) core classes and functions.
	Developed by Léo Cances (leocances on Github).

	Modified : Yes
		- typing & imports
"""

import numpy as np
import os
import random
import torch

from tqdm import trange
from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple

from sslh.datasets.gsc_core import SPEECHCOMMANDS


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


def _split_s_u(train_dataset: SPEECHCOMMANDS, s_ratio: float = 1.0):
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
	def __init__(
		self,
		root: str,
		subset: str = "train",
		url: str = URL,
		download: bool = False,
		transform: Optional[Module] = None
	) -> None:
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
			self.__getitem__.cache.step_update(disk_cache)
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

	def _load_item(self, filepath: str, path: str) -> Tuple[str, str, int]:
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

	def __getitem__(self, index: int) -> Tuple[int, int, int]:
		fileid = self._walker[index]

		label, speaker_id, utterance_number = self._load_item(fileid, self._path)

		return target_mapper[label], speaker_id, utterance_number
