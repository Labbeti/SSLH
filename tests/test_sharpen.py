import torch

from mixmatch.sharpen import sharpen_multi, sharpen


def test_1():
	distribution = torch.as_tensor([
		[0.1462, 0.0504, 0.0578, 0.1058, 0.0975, 0.1227, 0.0741, 0.1160, 0.1387, 0.0908]
	])
	print("Distribution:", distribution)

	result = sharpen(distribution, 0.5, dim=1)
	print("Result:", result)


def test_2():
	distribution = torch.as_tensor([
		[0.9, 0.4, 0.6],
		[0.1, 0.9, 0.9],
	])
	print("Distribution:", distribution)

	result = sharpen_multi(distribution, 0.5, 0.5)
	print("Result:", result)


if __name__ == "__main__":
	test_1()
