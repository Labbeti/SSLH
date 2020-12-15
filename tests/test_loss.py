
import torch

from ssl.utils.misc import reset_seed
from ssl.utils.torch import JSDivLoss, JSDivLossWithLogits, Entropy
from torch import Tensor
from time import time
from timeit import timeit
from torch.nn import Softmax
from unittest import TestCase, main


class TestJS(TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		seed = torch.randint(low=0, high=1000, size=[]).item()
		reset_seed(seed)
		print("Seed ", seed)

	def create_logits(self) -> (Tensor, Tensor):
		bsize = 128
		nb_classes = 10
		min_, max_ = -1, 1
		logits_p = torch.rand(bsize, nb_classes) * (max_ - min_) + min_
		logits_q = torch.rand(bsize, nb_classes) * (max_ - min_) + min_
		return logits_p, logits_q

	def crit_1(self):
		logits_p, logits_q = self.create_logits()
		criterion = JSDivLoss(reduction="mean")
		softmax = Softmax(dim=1)
		_ = criterion(softmax(logits_p), softmax(logits_q))

	def crit_2(self):
		logits_p, logits_q = self.create_logits()
		criterion = JSDivLossWithLogits(reduction="mean")
		_ = criterion(logits_p, logits_q)

	def test_js(self):
		logits_p, logits_q = self.create_logits()

		t = [0, 0]
		r = [0, 0]

		s1 = time()
		l2 = JSDivLoss(reduction="mean")
		softmax = Softmax(dim=1)
		r[0] += l2(softmax(logits_p), softmax(logits_q))
		t[0] += time() - s1

		s2 = time()
		l1 = JSDivLossWithLogits(reduction="mean")
		r[1] += l1(logits_p, logits_q)
		t[1] += time() - s2

		print("Durations")
		print(t[0])
		print(t[1])

		print("Sum Results")
		print(r[0])
		print(r[1])

	def test_time(self):
		print("TimeIt")
		print(timeit(self.crit_1, number=10000))
		print(timeit(self.crit_2, number=10000))


class TestEntropy(TestCase):
	def test_ent(self):
		ent = Entropy(reduction="none")

		d1 = [0.5, 0.5, 0, 0]
		d2 = [0, 1, 0, 0]
		d3 = [0.25, 0.25, 0.25, 0.25]
		d4 = [0.25, 0.5, 0.125, 0.125]

		dis = [d1, d2, d3, d4]

		res = ent(torch.as_tensor(dis))
		print(res)


if __name__ == "__main__":
	main()
