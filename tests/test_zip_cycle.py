
from mlu.utils.zip_cycle import ZipCycle


def test():
	r1 = range(1, 4)
	r2 = range(1, 6)
	iters = ZipCycle([r1, r2])
	for v1, v2 in iters:
		print(v1, v2)


if __name__ == "__main__":
	test()
