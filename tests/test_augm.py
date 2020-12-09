import os.path as osp
import unittest

from argparse import Namespace
from augmentation_utils.signal_augmentations import TimeStretch
from matplotlib import pyplot as plt

from sslh.augments.signal_augments import ResizePadCut
from sslh.augments.utils import Squeeze, PadUpTo
from sslh.datasets.gsc import GSCInterface

from torch import Tensor
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from unittest import TestCase


def create_args() -> Namespace:
	args = Namespace()
	args.dataset_path = osp.join("..", "dataset")
	args.label_smoothing_value = None
	return args


def to_spec(signal: Tensor) -> Tensor:
	spec = Sequential(
		PadUpTo(target_length=16000, mode="constant", value=0),
		MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
		AmplitudeToDB(),
		Squeeze(),
	)
	return spec(signal)


class TestGSCAugm(TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.itf = GSCInterface()

	def test_ds_strong(self):
		args = create_args()
		ds_strong = self.itf.get_dataset_train_augm_strong(args)
		print("Shape for", self.itf.get_dataset_name(), ":", ds_strong[0][0].shape, ";", ds_strong[0][1].shape)
		print(ds_strong[0])


class TestStretch(TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.itf = GSCInterface()

	def test_time_stretch(self):
		args = create_args()
		dataset = self.itf.get_dataset_train_with_transform(args, None, None)

		signal, label = dataset[0]
		spec = to_spec(signal)

		rate = (0.9, 0.9)
		stretch = TimeStretch(1.0, rate=rate)
		stretch_pad = ResizePadCut(1.0, rate=rate)

		print("Original signal shape: ", signal.shape)

		signal_stretched_1 = stretch(signal[0]).unsqueeze(dim=0)
		print("TimeStretch shape: ", signal_stretched_1.shape)

		signal_stretched_2 = stretch_pad(signal)
		print("TimeStretchPadCrop shape: ", signal_stretched_2.shape)

		# plt.figure()
		# plt.imshow(spec, origin="lower")
		plt.figure()
		plt.plot(signal[0])
		plt.figure()
		plt.plot(signal_stretched_1[0])
		plt.figure()
		plt.plot(signal_stretched_2[0])
		plt.show()


if __name__ == "__main__":
	unittest.main()
