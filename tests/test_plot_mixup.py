import json
import numpy as np
import os.path as osp

from matplotlib import pyplot as plt


def get_img(filepath: str) -> (np.ndarray, int):
	with open(filepath, "r") as file:
		data = json.load(file)
		x = data["x"]
		label = data["y"]
		return np.array(x, dtype=np.uint8), label


def main():
	filepath1 = osp.join("..", "results", "img_908.json")
	filepath2 = osp.join("..", "results", "img_8957.json")

	img1, label_idx1 = get_img(filepath1)
	img2, label_idx2 = get_img(filepath2)

	label1 = np.zeros(10)
	label1[label_idx1] = 1

	label2 = np.zeros(10)
	label2[label_idx2] = 1

	lambda_ = 0.6

	img_mixed = img1 * lambda_ + img2 * (1.0 - lambda_)
	label_mixed = label1 * lambda_ + label2 * (1.0 - lambda_)

	names = ["Img 1", "Img 2", "Img mixed"]
	imgs = [img1, img2, img_mixed]
	for name, img in zip(names, imgs):
		plt.figure()
		plt.title(name)
		plt.imshow(np.array(img, dtype=int))

	labels = [label1, label2, label_mixed]
	for label in labels:
		plt.figure()
		plt.bar(list(range(len(label))), label)
		axes = plt.gca()
		axes.set_ylim([0.0, 1.1])

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	main()
