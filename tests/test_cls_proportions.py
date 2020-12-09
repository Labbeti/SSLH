
import os.path as osp

from argparse import Namespace
from sslh.datasets.get_interface import get_dataset_interface
from sslh.datasets.utils import get_classes_idx
from matplotlib import pyplot as plt


def create_args() -> Namespace:
	args = Namespace()
	args.dataset = "GSC12"
	args.dataset_path = osp.join("..", "dataset")
	args.label_smoothing_value = None
	args.augm_none = None
	return args


def show():
	args = create_args()

	itf = get_dataset_interface(args.dataset)
	dataset_train = itf.get_dataset_train(args)
	cls_idx_all = get_classes_idx(dataset_train, itf.get_nb_classes(), is_one_hot=True)
	proportions = [len(cls_idx) for cls_idx in cls_idx_all]
	print(proportions)
	print(sum(proportions))
	print(len(dataset_train))
	# sum_ = sum(proportions)
	# proportions = [p / sum_ for p in proportions]

	name = "Proportions_{}".format(itf.get_dataset_name())
	plt.title(name)
	plt.bar(list(range(len(proportions))), proportions)
	plt.show()


if __name__ == "__main__":
	show()
