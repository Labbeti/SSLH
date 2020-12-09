import json
import matplotlib.pyplot as plt
import os.path as osp

from torchvision.datasets import CIFAR10
from sslh.augments.img_augments import *


ratio = 1.0
enhance_range = (1.0, 1.0)  # (0.05, 0.95)
# 14 callables
ra_augments_set = [
	AutoContrast(ratio=ratio),
	Brightness(ratio=ratio, levels=enhance_range),
	Color(ratio=ratio, levels=enhance_range),
	Contrast(ratio=ratio, levels=enhance_range),
	Equalize(ratio=ratio),
	Identity(),
	Posterize(ratio=ratio, nbs_bits=(4, 8)),
	Rotation(ratio=ratio, angles=(-30, 30)),
	Sharpness(ratio=ratio, levels=enhance_range),
	ShearX(ratio=ratio, shears=(-0.3, 0.3)),
	ShearY(ratio=ratio, shears=(-0.3, 0.3)),
	Solarize(ratio=ratio, thresholds=(0, 255)),
	TranslateX(ratio=ratio, deltas=(-0.3, 0.3)),
	TranslateY(ratio=ratio, deltas=(-0.3, 0.3)),
]
# 17 callables
cta_augments_set = [
	Blend(AutoContrast(ratio=ratio), levels=(0, 1)),
	Brightness(ratio=ratio, levels=(0, 1)),
	Color(ratio=ratio, levels=(0, 1)),
	Contrast(ratio=ratio, levels=(0, 1)),
	Blend(Equalize(ratio=ratio), levels=(0, 1)),
	Blend(Invert(ratio=ratio), levels=(0, 1)),
	Identity(),
	Posterize(ratio=ratio, nbs_bits=(1, 8)),
	Rescale(ratio=ratio),
	Rotation(ratio=ratio, angles=(-30, 30)),
	Sharpness(ratio=ratio, levels=(0, 1)),
	ShearX(ratio=ratio, shears=(-0.3, 0.3)),
	ShearY(ratio=ratio, shears=(-0.3, 0.3)),
	Blend(Smooth(ratio=ratio), levels=(0, 1)),
	Solarize(ratio=ratio, thresholds=(0, 255)),
	TranslateX(ratio=ratio, deltas=(-0.3, 0.3)),
	TranslateY(ratio=ratio, deltas=(-0.3, 0.3)),
]


def get_demo_image() -> np.ndarray:
	img = np.zeros((3, 128, 128), dtype=np.uint8)
	img[0] = np.linspace(start=list(range(128)), stop=list(range(128, 256)), num=128)
	img[1] = np.linspace(start=list(reversed(range(128))), stop=list(reversed(range(128, 256))), num=128)
	img[2, 0:16] = 128
	return img.T


def get_saved_img() -> (np.ndarray, int):
	idx = 908  # 908, 8957
	filepath = "../results/img/img_{:d}.json".format(idx)
	with open(filepath, "r") as file:
		data = json.load(file)
		x = data["x"]
		return np.array(x, dtype=np.uint8), idx


def save_cifar_img():
	dataset_path = osp.join("..", "dataset")
	dataset = CIFAR10(dataset_path, train=False, download=False, transform=np.array)

	idx = np.random.randint(0, len(dataset))
	idx = 908
	img, label = dataset[idx]
	data = {"x": img.tolist(), "y": label, "index": idx}
	filepath = "../results/img/img_{:d}.json".format(idx)
	with open(filepath, "w") as file:
		json.dump(data, file, indent="\t")


def test():
	# save_cifar_img()
	img, idx = get_saved_img()

	augms = [
		Identity(),
		# Invert(ratio=1.0),
		# HorizontalFlip(ratio=1.0),
		# AutoContrast(),
		Rotation(ratio=ratio, angles=(0, 0)),
		Rotation(ratio=ratio, angles=(90, 90)),
		Rotation(ratio=ratio, angles=(180, 180)),
		Rotation(ratio=ratio, angles=(270, 270)),
		# Inversion(),
		# UniColor(),
		# CutOut(),
		# Gray(),
		# Posterize(ratio=ratio, nbs_bits=(3, 4)),
	]
	print("Img original shape = {:s} (type={:s})".format(img.shape, img.dtype))

	dirpath = osp.join("..", "results", "img")
	prefix = "img"

	for augm in augms:
		img_a = augm(img.copy())
		print("Img augm shape = {:s} (type={:s})".format(img_a.shape, img_a.dtype))

		name = augm.__class__.__name__

		fig = plt.figure(frameon=False)
		plt.title(name)
		plt.imshow(np.array(img_a, dtype=int))
		filepath = osp.join(dirpath, "{:s}_{:s}.png".format(prefix, name))
		i = 2
		while osp.isfile(filepath):
			filepath = osp.join(dirpath, "{:s}_{:s}_{:d}.png".format(prefix, name, i))
			i += 1

		fig.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	test()
