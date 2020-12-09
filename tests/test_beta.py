
from matplotlib import pyplot as plt
from typing import List


def beta_pdf(x: List[float], alpha: float, beta: float) -> (List[float], List[float]):
	new_x = [u for u in x if 0.0 < u < 1.0]
	new_y = [u ** (alpha - 1.0) * (1 - u) ** (beta - 1.0) for u in new_x]
	sum_ = sum(new_y)
	new_y = [v / sum_ for v in new_y]
	return new_x, new_y


def plot_beta_pdf():
	nb_points = 10000
	x = list(range(1, nb_points+1))
	x = [v / nb_points for v in x]

	hparams = [
		(0.4, 0.4),
		(0.75, .75),
		(1.0, 1.0),
		(2.0, 2.0),
		(1.5, 1.5),
	]
	min_, max_ = 0, 0.0003
	for alpha, beta in hparams:
		x, y = beta_pdf(x, alpha, beta)
		# Filter values
		x_new, y_new = [], []
		for v1, v2 in zip(x, y):
			if min_ <= v2 <= max_:
				x_new.append(v1)
				y_new.append(v2)
		# Plot
		plt.plot(x_new, y_new, label="beta_{:.2f}_{:.2f}".format(alpha, beta))

	# plt.yscale("log")
	plt.legend()
	plt.show()


if __name__ == "__main__":
	plot_beta_pdf()
