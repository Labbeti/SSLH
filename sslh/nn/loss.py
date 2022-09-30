#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from torch import nn, Tensor


DEFAULT_EPSILON = 2e-20


class NLLLossVecTargets(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        dim: Optional[int] = -1,
        log_input: bool = False,
        pred_clamp_min: float = DEFAULT_EPSILON,
    ) -> None:
        """
        Compute NLLLoss between two distributions probabilities.

        Input and targets must be a batch of probabilities distributions of shape (batch_size, n_classes) tensor.
        Useful when target is not a one-hot label, like in Label-smoothing or MixMatch methods.
        """
        super().__init__()
        self.reduction = reduction
        self.dim = dim
        self.log_input = log_input
        self.pred_clamp_min = pred_clamp_min
        self.reduce_fn = get_reduction_from_name(reduction)

    def forward(self, pred: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy with targets.
        Input and target must be a (batch_size, n_classes) tensor.
        """
        if pred.shape != targets.shape:
            raise ValueError(
                f"Invalid arguments shapes {pred.shape=} and {targets.shape=}. (expected  the same shape for pred and targets)"
            )

        if not self.log_input:
            pred = torch.clamp(pred, min=self.pred_clamp_min)
            pred = torch.log(pred)
        loss = -torch.sum(pred * targets, dim=self.dim)  # type: ignore
        loss = self.reduce_fn(loss)
        return loss

    def extra_repr(self) -> str:
        return f"reduce_fn={self.reduce_fn.__name__}, dim={self.dim}, log_input={self.log_input}, pred_clamp_min={self.pred_clamp_min}"


class Entropy(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        dim: int = -1,
        epsilon: float = DEFAULT_EPSILON,
        base: Optional[float] = None,
        log_input: bool = False,
    ) -> None:
        """
        Compute the entropy of a distribution.

        :param reduction: The reduction used between batch entropies. (default: 'mean')
        :param dim: The dimension to apply the sum in entropy formula. (default: -1)
        :param epsilon: The epsilon precision to use. Must be a small positive float. (default: DEFAULT_EPSILON)
        :param base: The log-base used. If None, use the natural logarithm (i.e. base = torch.exp(1)). (default: None)
        :param log_input: If True, the input must be log-probabilities. (default: False)
        """
        super().__init__()
        self.reduce_fn = get_reduction_from_name(reduction)
        self.dim = dim
        self.epsilon = epsilon
        self.log_input = log_input

        if base is None:
            self.log_func = torch.log
        else:
            log_base = torch.log(torch.scalar_tensor(base))
            self.log_func = lambda x: torch.log(x) / log_base

    def forward(self, probabilities: Tensor) -> Tensor:
        if not self.log_input:
            entropy = -torch.sum(
                probabilities * self.log_func(probabilities + self.epsilon),
                dim=self.dim,
            )
        else:
            entropy = -torch.sum(torch.exp(probabilities) * probabilities, dim=self.dim)
        return self.reduce_fn(entropy)

    def extra_repr(self) -> str:
        return f"reduce_fn={self.reduce_fn.__name__}, dim={self.dim}, epsilon={self.epsilon}, log_input={self.log_input}"


class JSDivLoss(nn.Module):
    def __init__(
        self, reduction: str = "mean", dim: int = -1, epsilon: float = DEFAULT_EPSILON
    ):
        """
        Jensen-Shannon Divergence loss.

        Use the following formula :

        >>> 'JS(p,q) = H(0.5 * (p+q)) - 0.5 * (H(p) + H(q))'

        :param reduction: The reduction function to apply. (default: 'mean')
        :param dim: The dimension of the probabilities. (default: -1)
        :param epsilon: The epsilon value used for numerical stability. (default: DEFAULT_EPSILON)
        """
        super().__init__()
        self.entropy = Entropy(reduction, dim, epsilon, log_input=False)

    def forward(self, p: Tensor, q: Tensor) -> Tensor:
        a = self.entropy(0.5 * (p + q))
        b = 0.5 * (self.entropy(p) + self.entropy(q))
        return a - b


class JSDivLossFromLogits(nn.Module):
    def __init__(
        self, reduction: str = "mean", log_activation: nn.Module = nn.LogSoftmax(dim=-1)
    ):
        """
        Jensen-Shannon Divergence loss with logits.

        Use the following formula :

        >>> 'JS(p,q) = 0.5 * (KL(LS(p),m) + KL(LS(q),m)), with m = LS(0.5 * (p+q))'
        >>> 'where LS = LogSoftmax and KL = KL-Divergence.'

        :param reduction: The reduction function to apply. (default: 'mean')
        :param log_activation: The log-activation function for compute predictions from logits. (default: LogSoftmax(dim=-1))
        """
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.log_activation = log_activation

    def forward(self, logits_p: Tensor, logits_q: Tensor) -> Tensor:
        m = self.log_activation(0.5 * (logits_p + logits_q))
        p = self.log_activation(logits_p)
        q = self.log_activation(logits_q)

        a = self.kl_div(p, m)
        b = self.kl_div(q, m)

        return 0.5 * (a + b)


class KLDivLossWithProbabilities(nn.KLDivLoss):
    def __init__(
        self,
        reduction: str = "mean",
        epsilon: float = DEFAULT_EPSILON,
        log_input: bool = False,
        log_target: bool = False,
    ):
        """
        KL-divergence with probabilities.

        :param reduction: The reduction function to apply. (default: 'mean')
        :param epsilon: The epsilon value used for numerical stability. (default: DEFAULT_EPSILON)
        :param log_input: If True, the probabilities of the first argument are transform to log scale internally.
        :param log_target:

        """
        super().__init__(reduction=reduction, log_target=log_target)
        self.epsilon = epsilon
        self.log_input = log_input
        self.log_target = log_target

    def forward(self, p: Tensor, q: Tensor) -> Tensor:
        if not self.log_input:
            p = torch.log(p + torch.scalar_tensor(self.epsilon))
        return super().forward(input=p, target=q)

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}, log_input={self.log_input}, log_target={self.log_target}"


class BCELossBatchMean(nn.Module):
    def __init__(self, reduce_fn: Callable = torch.mean, dim: int = -1):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.reduce_fn = reduce_fn
        self.dim = dim

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.reduce_fn(self.bce(pred, target), dim=self.dim)


class BCELossSoftMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        n_targets = target.sum(dim=-1).clamp(min=1.0)
        return self.bce(pred, target).sum(dim=-1) / n_targets


def identity(x: Tensor) -> Tensor:
    return x


def batchmean(x: Tensor) -> Tensor:
    return torch.mean(x, dim=-1)


def get_reduction_from_name(name: str) -> Callable[[Tensor], Tensor]:
    """
    :param name: The name of the reduction function.
            Available functions are 'sum' and 'mean', 'none' and 'batchmean'.
    :return: The reduction function with a name.
    """
    if name in ["mean"]:
        return torch.mean
    elif name in ["sum"]:
        return torch.sum
    elif name in ["none", "identity"]:
        return identity
    elif name in ["batchmean"]:
        return batchmean
    else:
        raise RuntimeError(
            f'Unknown reduction "{name}". Must be one of {str(["mean", "sum", "none", "batchmean"])}.'
        )
