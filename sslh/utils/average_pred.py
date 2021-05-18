import torch

from torch import Tensor


class AveragePred:
	def __init__(self, history: int = 128):
		self.history = history
		self.preds = []
		self.cur_idx = 0

	def reset(self):
		"""
			Reset the history.
		"""
		self.preds = []
		self.cur_idx = 0

	def add_pred(self, pred: Tensor):
		"""
			Add a batch of predictions of shape (bsize, n_classes) to the history for computing the classes distributions.
		"""
		if len(self.preds) >= self.history:
			self.preds[self.cur_idx] = pred
			self.cur_idx = (self.cur_idx + 1) % self.history
		else:
			self.preds.append(pred)

	def get_mean(self) -> Tensor:
		"""
			Compute the mean of the predictions stored, i.e. an approximation of the classes distribution.
		"""
		return torch.stack(self.preds).mean(dim=(0, 1))
