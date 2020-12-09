import numpy as np
from matplotlib import pyplot as plt


def main():
	values = np.asarray([0.6, 0.1, 0.1, 0.2])
	ps = np.asarray([0.15, 0.55, 0.15, 0.15])
	pu = np.asarray([0.25, 0.25, 0.25, 0.25])

	new_values = values / pu * ps
	new_values = new_values / sum(new_values)

	names = ["Original distribution", "Supervised distribution", "Unsupervised distribution", "New distribution"]
	distributions = [values, ps, pu, new_values]

	for name, distribution in zip(names, distributions):
		plt.figure()
		plt.title(name)
		plt.bar(list(range(len(distribution))), distribution)

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	main()
