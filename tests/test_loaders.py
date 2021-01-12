
from mlu.datasets.wrappers import ZipDataset
from mlu.utils.zip_cycle import ZipCycle

from mlu.datasets.utils import _get_classes_idx, _shuffle_classes_idx, _split_classes_idx

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class DummyDataset(Dataset):
	def __init__(self, label: int = 0):
		self.label = label

	def __getitem__(self, item):
		return item, self.label

	def __len__(self):
		return 10


def test_1():
	ds = DummyDataset()
	nb_classes = 5
	ratios = [0.2, 0.8]

	cls_idx_all = _get_classes_idx(ds, nb_classes)
	cls_idx_all = _shuffle_classes_idx(cls_idx_all)
	idx_train = _split_classes_idx(cls_idx_all, ratios)

	loader_0 = DataLoader(ds, batch_size=2, drop_last=True, sampler=SubsetRandomSampler(idx_train[0]))
	loader_1 = DataLoader(ds, batch_size=3, drop_last=False, sampler=SubsetRandomSampler(idx_train[1]))
	loader = ZipCycle([loader_0, loader_1], policy="max")

	for items in loader:
		print("Items: ", items)


def test_2():
	ds1 = DummyDataset()
	ds2 = DummyDataset(-1)
	ds3 = ZipDataset([ds1, ds1])
	ds = ZipDataset([ds3, ds2])

	loader = DataLoader(ds, batch_size=4)
	for items in loader:
		print("Items: ", items)


if __name__ == "__main__":
	# test_1()
	test_2()
