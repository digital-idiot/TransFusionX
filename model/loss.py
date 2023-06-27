import torch
from typing import Optional
from einops import rearrange
from torch.nn.functional import one_hot
from torchmetrics.functional import recall
from torch.nn.functional import cross_entropy


# noinspection PyShadowingBuiltins
class DiceLoss(torch.nn.Module):
    def __init__(
            self,
            num_classes: Optional[int] = None,
            prediction_logit: bool = True,
            smooth: float = 1.0
    ):
        super(DiceLoss, self).__init__()
        self._logit = prediction_logit
        self._n_classes = num_classes
        self._smooth = smooth

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ):
        if self._logit:
            input = input.softmax(dim=1)
        input = rearrange(input, 'b c ... -> b c (...)')
        if isinstance(
            self._n_classes, int
        ) and (
            self._n_classes > 0
        ):
            target = rearrange(
                one_hot(
                    target,
                    num_classes=self._n_classes
                ),
                'b ... c -> b c (...)'
            )

        target = target.to(input)

        numerator = self._smooth + torch.sum(
            (2 * input * target), dim=-1
        )

        denominator = self._smooth + torch.sum(
            (input ** 2), dim=-1
        ) + torch.sum(
            (target ** 2), dim=-1
        )
        dice = numerator / denominator
        return (1 - dice).mean()


class OhemCrossEntropyLoss(torch.nn.Module):
    def __init__(
            self,
            num_classes: Optional[int] = None,
            ignore_index: int = None,
            threshold: float = 0.7,
            min_kept: float = 0.1
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.ignore_index = ignore_index
        assert 0 <= min_kept <= 1.0, (
            f"In valid 'min_kept' ratio: {min_kept}.\n" +
            "Valid range: [0.0, 1.0]"
        )
        self.min_kept = min_kept

    def ohem(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.detach().clone().softmax(dim=1)
        target = target.detach().clone()
        valid_mask = torch.full_like(
            target,
            fill_value=True,
            dtype=torch.bool,
            device=target.device
        ) if (self.ignore_index is None) else (target != self.ignore_index)
        num_valid = valid_mask[valid_mask].numel()
        if num_valid > 0:
            if self.num_classes is not None:
                target = rearrange(
                    one_hot(
                        target,
                        num_classes=self.num_classes
                    ),
                    'b ... c -> b c (...)'
                )
            prob = ((pred * target).sum(dim=1, keepdim=False))
            valid_prob = prob[valid_mask]
            min_kept = int(self.min_kept * valid_prob.numel())
            if 0 < min_kept < num_valid:
                index = valid_prob.argsort(dim=-1)
                threshold_index = index[min(index.numel(), min_kept) - 1]
                self.threshold = max(
                    valid_prob[threshold_index], self.threshold
                )
                valid_mask = torch.logical_and(
                    input=valid_mask,
                    other=(prob < self.threshold)
                )
        return valid_mask

    # noinspection SpellCheckingInspection,PyShadowingBuiltins
    def forward(
            self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # get the label after ohem
        input = rearrange(input, 'b c ... -> b c (...)')
        target = rearrange(target, "b ... -> b (...)")
        valid_mask = self.ohem(pred=input, target=target)
        target[torch.logical_not(valid_mask)] = self.ignore_index
        loss = cross_entropy(
            input=input,
            target=target,
            weight=None,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=0.0
        )
        return loss[valid_mask].mean()


class RecallCrossEntropyLoss(torch.nn.Module):
    def __init__(
            self,
            num_classes: Optional[int] = None,
            ignore_index: int = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    # noinspection SpellCheckingInspection,PyShadowingBuiltins
    def forward(
            self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # get the label after ohem
        input = rearrange(input, 'b c ... -> b c (...)').softmax(dim=1)
        target = rearrange(target, "b ... -> b (...)")
        weights = 1 - recall(
            preds=input,
            target=target,
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_classes,
            average="micro",
            multidim_average="samplewise",
            top_k=1,
            ignore_index=self.ignore_index,
            validate_args=False
        )
        rce = -1 * weights * torch.log(
            input.gather(dim=1, index=target.unsqueeze(dim=1)).squeeze(dim=1)
        ).mean(dim=-1)
        return rce.mean()
