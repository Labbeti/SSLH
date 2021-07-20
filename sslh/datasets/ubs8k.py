import os
import os.path as osp
import torchaudio

from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.transforms import Resample
from typing import Callable, Dict, List, Optional, Tuple


class URBANSOUND8K(Dataset):
	ROOT_DNAME: str = 'UrbanSound8K'
	N_CLASSES: int = 10

	def __init__(self, root: str, folds: List[int], resample_sr: int = 22050) -> None:
		"""
			UBS8K dataset.

			:param root: The directory path to the dataset root.
			:param folds: The folds to use.
			:param resample_sr: The resample rate of the waveform.
		"""
		super().__init__()
		self.root = root
		self.folds = list(folds)
		self.resample_sr = resample_sr

		self.meta = self._load_metadata()
		self.wav_dir = os.path.join(root, self.ROOT_DNAME, 'audio')

	def __getitem__(self, idx) -> Tuple[Tensor, int]:
		filename = self.meta['filename'][idx]
		target = self.meta['target'][idx]
		fold = self.meta['fold'][idx]

		file_path = os.path.join(self.wav_dir, f'fold{fold}', filename)

		waveform, sr = torchaudio.load(file_path)
		waveform = self._to_mono(waveform)
		waveform = self._resample(waveform, sr)
		waveform = waveform.squeeze()

		return waveform, target

	def __len__(self) -> int:
		return len(self.meta['filename'])

	def _load_metadata(self) -> Dict[str, list]:
		csv_path = os.path.join(self.root, self.ROOT_DNAME, 'metadata', 'UrbanSound8K.csv')

		with open(csv_path) as file:
			lines = file.read().splitlines()
			lines = lines[1:]  # remove the header

		info = {'filename': [], 'fold': [], 'target': []}
		for line in lines:
			line = line.split(',')
			if int(line[5]) in self.folds:  # l[6] == file folds
				info['filename'].append(line[0])
				info['fold'].append(int(line[5]))
				info['target'].append(int(line[6]))

		return info

	def _resample(self, waveform: Tensor, sr: int) -> Tensor:
		resampler = Resample(sr, self.resample_sr)
		return resampler(waveform)

	def _to_mono(self, waveform: Tensor) -> Tensor:
		if len(waveform.shape) == 2:
			if waveform.shape[0] == 1:
				return waveform
			else:
				return waveform.mean(dim=0)
		else:
			raise ValueError(
				f'waveform tensor should be of shape (channels, time). currently is of shape {waveform.shape}'
			)


class UBS8KDataset(URBANSOUND8K):
	def __init__(self, root: str, folds: List[int], transform: Optional[Callable] = None, cached: bool = False):
		super().__init__(osp.dirname(root), folds)
		self.transform = transform
		self.cached = cached
		self._waveform_cache = {}

	def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
		if idx in self._waveform_cache.keys():
			waveform = self._waveform_cache[idx]
			target = self.meta['target'][idx]
		else:
			waveform, target = super().__getitem__(idx)
			if self.cached:
				self._waveform_cache[idx] = waveform

		if self.transform is not None:
			waveform = self.transform(waveform)

		return waveform, target
