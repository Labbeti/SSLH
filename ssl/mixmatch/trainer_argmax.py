
import torch

from ssl.mixmatch.trainer import MixMatchTrainer
from torch import Tensor
from torch.nn.functional import one_hot


class MixMatchTrainerArgmax(MixMatchTrainer):
	def guess_label(self, batch_u_augm_weak_multiple: Tensor, temperature: float) -> Tensor:
		nb_augms = batch_u_augm_weak_multiple.shape[0]
		preds = [torch.zeros(0) for _ in range(nb_augms)]
		for k in range(nb_augms):
			logits = self.model(batch_u_augm_weak_multiple[k])
			preds[k] = self.activation(logits, dim=1)
		preds = torch.stack(preds)
		labels_u = preds.mean(dim=0)

		nb_classes = labels_u.shape[1]
		labels_u = one_hot(labels_u.argmax(dim=1), nb_classes)

		return labels_u
