import torch

from ssl.utils.torch import normalized
from torch import Tensor
from torch.nn import Module


class Sharpen(Module):
	"""
		One-hot sharpening class used in MixMatch and ReMixMatch.
	"""
	def __init__(self, temperature: float):
		super().__init__()
		self.temperature = temperature

	def forward(self, batch: Tensor, dim: int) -> Tensor:
		return sharpen(batch, self.temperature, dim)


class SharpenMulti(Module):
	"""
		Experimental multi-hot sharpening class.
	"""
	def __init__(self, temperature: float, threshold: float):
		super().__init__()
		self.temperature = temperature
		self.threshold = threshold

	def forward(self, batch: Tensor, dim: int) -> Tensor:
		return sharpen_multi(batch, self.temperature, self.threshold)


def sharpen(batch: Tensor, temperature: float, dim: int) -> Tensor:
	"""
		Sharpen function. Increase the higher probability of a distribution and decrease the others.
			p_i = p_i ** (1 / temperature) / sum_j p_j ** (1 / temperature)

		:param batch: A tensor of shape (bsize, nb_classes) or (nb_classes,) of the distributions.
		:param temperature: The sharpening temperature used in range ]0.0, 1.0].
			If temperature -> 0, make the distribution more "one-hot".
			If temperature -> 1, the sharpen does not change the distribution.
		:param dim: The dimension to apply the sharpening.
		:returns: The distribution sharpened.
	"""
	batch = batch ** (1.0 / temperature)
	return normalized(batch, dim=dim)


def sharpen_multi(batch: Tensor, temperature: float, threshold: float) -> Tensor:
	"""
		Experimental multi-hot sharpening function.
		:param batch: The batch to sharpen.
		:param temperature: Temperature of the sharpen function.
		:param threshold: Threshold used to determine if a probability must be increased or decreased.
		:returns: The batch sharpened.
	"""
	result = batch.clone()
	nb_dim = len(batch.shape)

	if nb_dim == 1:
		return _sharpen_multi_experiment_2(batch, temperature, threshold)
	elif nb_dim == 2:
		for i, distribution in enumerate(batch):
			result[i] = _sharpen_multi_experiment_2(distribution, temperature, threshold)
	elif nb_dim == 3:
		for i, distribution_i in enumerate(batch):
			for j, distribution_j in enumerate(distribution_i):
				result[i, j] = _sharpen_multi_experiment_2(distribution_j, temperature, threshold)
	else:
		raise RuntimeError("Invalid nb_dim {:d}. (only 1, 2 or 3)".format(nb_dim))

	return result


def _sharpen_multi_experiment_1(distribution: Tensor, temperature: float, threshold: float) -> Tensor:
	""" Experimental V1 multi-hot sharpening. Currently unused. """
	k = (distribution > threshold).long().sum().item()
	if k < 1:
		return distribution

	sorted_, idx = distribution.sort(descending=True)
	preds, others = sorted_[:k], sorted_[k:]
	preds_idx, others_idx = idx[:k], idx[k:]

	tmp = torch.zeros((len(preds), 1 + len(others)))
	for i, v in enumerate(preds):
		sub_distribution = torch.cat((v.unsqueeze(dim=0), others))
		sub_distribution = sharpen(sub_distribution, temperature, dim=0)
		tmp[i] = sub_distribution

	new_dis = torch.zeros(distribution.size())
	new_dis[preds_idx] = tmp[:, 0].squeeze()
	new_dis[others_idx] = tmp[:, 1:].mean(dim=0)
	return new_dis


def _sharpen_multi_experiment_2(distribution: Tensor, temperature: float, threshold: float) -> Tensor:
	""" Experimental V2 multi-hot sharpening. """
	original_mask = (distribution > threshold).float()
	nb_above = original_mask.sum().long().item()

	if nb_above == 0:
		return distribution

	distribution_expanded = distribution.expand(nb_above, *distribution.shape).clone()
	mask_nums = original_mask.argsort(descending=True)[:nb_above]

	mask_nums_expanded = torch.zeros(*distribution.shape[:-1], nb_above, nb_above - 1).long()
	for i in range(nb_above):
		indices = list(range(nb_above))
		indices.remove(i)
		mask_nums_expanded[i] = mask_nums[indices].clone()

	for i, (distribution, nums) in enumerate(zip(distribution_expanded, mask_nums_expanded)):
		distribution_expanded[i][nums] = 0.0

	distribution_expanded = sharpen(distribution_expanded, temperature, dim=1)

	result = distribution_expanded.mean(dim=0)
	result[mask_nums] = distribution_expanded.max(dim=1)[0]

	return result


def _sharpen_multi_experiment_3(batch: Tensor, temperature: float = 10.0, threshold: float = 0.5) -> Tensor:
	batch = torch.sigmoid((batch - threshold) * temperature)
	return batch
