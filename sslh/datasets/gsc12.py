"""
	Google Speech Commands Task (GSC12) core classes and functions.
	Developed by Léo Cances (leocances on Github).

	Modified : Yes
		- typing & imports
"""

import numpy as np
import os
import soundfile
import torchaudio

from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple

from sslh.datasets.gsc import SpeechCommands, URL, all_classes, cache_feature, EXCEPT_FOLDER


class SpeechCommand10(SpeechCommands):
	TRUE_CLASSES = ['yes', 'no', 'up', 'down', 'left',
					'right', 'off', 'on', 'go', 'stop']

	def __init__(
		self,
		root: str,
		subset: str = 'train',
		url: str = URL,
		download: bool = False,
		transform: Optional[Module] = None,
		percent_to_drop: float = 0.5
	) -> None:
		super().__init__(root, subset, url, download, transform)

		assert 0.0 <= percent_to_drop < 1.0

		self.percent_to_drop = percent_to_drop

		self.target_mapper = {
			'yes': 0,
			'no': 1,
			'up': 2,
			'down': 3,
			'left': 4,
			'right': 5,
			'off': 6,
			'on': 7,
			'go': 8,
			'stop': 9,
			'silence': 10,
			'unknown': 11  # _background_noise_
		}

		# the rest of the classes belong to the 'junk / trash / poubelle class'
		for cmd in all_classes:
			if cmd not in self.target_mapper:
				self.target_mapper[cmd] = 11

		self.drop_some_trash()
		self.add_silence()

	@cache_feature
	def __getitem__(self, index: int) -> Tuple[Tensor, int]:
		filepath = self._walker[index]

		label = filepath.split('/')[-2]
		target = self.target_mapper[label]

		waveform, _ = super().__getitem__(index)

		return waveform, target

	def drop_some_trash(self):
		def is_trash(path: str) -> bool:
			return self.target_mapper[path.split('/')[-2]] == 11

		# Create the complete list of trash class
		trash_list = [path for path in self._walker if is_trash(path)]

		# choice only x% of it that will be removed
		n_to_drop = int(len(trash_list) * self.percent_to_drop)
		to_drop = np.random.choice(trash_list, size=n_to_drop, replace=False)

		# remove it from the _walker
		self._walker = list(set(self._walker) - set(to_drop))

		print('%d out of %s junk files were drop.' % (len(to_drop), len(trash_list)))

	def add_silence(self):
		"""For the class silence, a new directory is created called 'silence'
		It will contain 1 seconds segment from the _background_noise_ directory
		If the directory already exist, do some verification and pass
		"""
		silence_dir = os.path.join(*self.root_path, 'silence')

		if os.path.isdir(silence_dir):
			self._check_silence_class()

		else:
			self._create_silence_class()
			self._check_silence_class()

	def _create_silence_class2(self):
		print('Silence class doesn\'t exist')
		silence_dir = os.path.join(*self.root_path, 'silence')
		noise_path = os.path.join(*self.root_path, '_background_noise_')

		# the silence class directory doesn't exist, create it
		os.makedirs(silence_dir)

		# Split each noise files into 1 second long segment
		to_process = []
		for file in os.listdir(os.path.join(*self.root_path, EXCEPT_FOLDER)):
			file: str
			if file[-4:] == '.wav':
				to_process.append(os.path.join(noise_path, file))

		# Basic way, split each files into 1 seconds long segment
		print('Creating silence samples...')
		for filepath in to_process:
			basename = os.path.basename(filepath)

			waveform, sr = torchaudio.load(filepath)

			n_full_segment = int(len(waveform[0]) / sr)
			rest = len(waveform[0]) % sr
			segments = np.split(waveform[0][:-rest], n_full_segment)

			# write each segment as a wav file with a unique name
			for i, s in enumerate(segments):
				unique_id = f'{basename[:-4]}_nohash_{i}.wav'
				path = os.path.join(silence_dir, unique_id)
				soundfile.write(path, s, sr)
		print('done')

	def _create_silence_class(self):
		print('Silence class doesn\'t exist')
		silence_dir = os.path.join(*self.root_path, 'silence')
		noise_path = os.path.join(*self.root_path, '_background_noise_')

		# the silence class directory doesn't exist, create it
		os.makedirs(silence_dir)

		# Split each noise files into 1 second long segment
		to_process = []
		for file in os.listdir(os.path.join(*self.root_path, EXCEPT_FOLDER)):
			file: str
			if file[-4:] == '.wav':
				to_process.append(os.path.join(noise_path, file))

		# Basic way, split each files into 1 seconds long segment
		print('Creating silence samples...')
		for filepath in to_process:
			basename = os.path.basename(filepath)

			waveform, sr = torchaudio.load(filepath)

			# write each segment, we will create 300 segments of 1 secondes
			start_timestamps = np.random.randint(0, len(waveform[0]) - sr, size=400)
			for i, st in enumerate(start_timestamps):
				unique_id = f'{basename[:-4]}_nohash_{i}.wav'
				path = os.path.join(silence_dir, unique_id)

				segment = waveform[0][st:st + sr]
				soundfile.write(path, segment, sr)

		print('done')

	def _check_silence_class(self):
		silence_dir = os.path.join(*self.root_path, 'silence')
		all_files = os.listdir(silence_dir)

		print('Silence class already processed')
		print('%s samples present' % len(all_files))
