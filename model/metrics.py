from typing import Optional
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassJaccardIndex


class MultiClassSegmentationMetrics(MetricCollection):
    def __init__(
            self,
            num_classes=int,
            ignore_index: Optional[int] = None,
            average: Optional[str] = 'macro',
            prefix: Optional[str] = None,
            postfix: Optional[str] = None,
            # True after fixing lightning 13254
            compute_groups: Optional[bool] = False,
    ):
        metrics_dict = {
            'Precision': MulticlassPrecision(
                num_classes=num_classes,
                multidim_average='global',
                ignore_index=ignore_index,
                average=average,
                top_k=1
            ),
            'Recall': MulticlassRecall(
                num_classes=num_classes,
                multidim_average='global',
                ignore_index=ignore_index,
                average=average,
                top_k=1
            ),
            'IoU': MulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average=average
            ),
            'F1': MulticlassF1Score(
                num_classes=num_classes,
                average=average,
                multidim_average='global',
                ignore_index=ignore_index,
                top_k=1
            ),
            'Accuracy': MulticlassAccuracy(
                num_classes=num_classes,
                average=average,
                multidim_average='global',
                ignore_index=ignore_index,
                top_k=1
            )
        }
        super(MultiClassSegmentationMetrics, self).__init__(
            metrics=metrics_dict,
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups
        )
