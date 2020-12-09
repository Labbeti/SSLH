import json
import numpy as np
from matplotlib import pyplot as plt


def main():
	filepath = "../results/signal_spec.json"
	with open(filepath, "r") as file:
		data = json.load(file)
		signal, spec = data["signal"], data["spec"]

	plt.figure()
	plt.plot(signal)
	plt.title("Signal")

	plt.figure()
	plt.title("Spectrogramme")
	plt.imshow(spec, origin="lower")

	plt.show(block=False)
	input("Press ENTER to quit\n> ")


if __name__ == "__main__":
	main()
