from sslh.mixmatch.warmup import WarmUp


def test():
	warmup = WarmUp(1.0, 10)
	for i in range(15):
		print("Value:", warmup.value())
		warmup.step()


if __name__ == "__main__":
	test()
