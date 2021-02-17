
import os.path as osp

from argparse import Namespace
from mlu.transforms.waveform import StretchPadCrop
from sslh.datasets.get_builder import get_dataset_builder


def create_args() -> Namespace:
	args = Namespace()
	args.dataset_path = osp.join("..", "datasets")
	args.augm_none = "weak"
	args.label_smoothing_value = None
	return args


def main():
	args = create_args()
	itf = get_dataset_builder("ESC10")
	augm = StretchPadCrop(rate=(0.5, 1.5), align="random", p=1.0)
	dataset = itf.get_dataset_train_with_transform(args, None, augm)

	for i in range(len(dataset)):
		print(dataset[i][0].shape)


if __name__ == "__main__":
	main()
