
import os.path as osp

from argparse import Namespace
from matplotlib import pyplot as plt
from mlu.datasets.utils import _get_classes_idx
from sslh.datasets.get_builder import get_dataset_builder


def create_args() -> Namespace:
	args = Namespace()
	args.dataset = "GSC12"
	args.dataset_path = osp.join("..", "datasets")
	args.label_smoothing_value = None
	args.augm_none = None
	return args


def show():
	args = create_args()

	itf = get_dataset_builder(args.dataset)
	dataset_train = itf.get_dataset_train(args.dataset_path)
	cls_idx_all = _get_classes_idx(dataset_train, itf.get_nb_classes(), target_one_hot=True)
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
